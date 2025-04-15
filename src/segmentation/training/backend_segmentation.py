import os
import time
import ujson
import logging

import numpy as np
import pandas as pd
from pathlib import Path
from contextlib import nullcontext

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torchinfo import summary
from torch.optim.lr_scheduler import LinearLR
from timm.scheduler import create_scheduler_v2

from deepspeed import initialize
from deepspeed.ops.adam import FusedAdam
from deepspeed.ops.lamb import FusedLamb
from deepspeed.runtime.lr_schedules import WarmupCosineLR

import ray.train.torch as raytorch
from ray.train import Checkpoint, report, get_context, get_checkpoint

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from segmentation.utils.comm import inference_context
from segmentation.data.gather_dataset import gather_dataset
from segmentation.checkpoint.checkpoint import load_checkpoint
from segmentation.training.registry import build_dependency_graph_and_instantiate

logger = logging.getLogger("ray")
logger.setLevel(logging.DEBUG)

dtypes = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def is_main_process():
    return get_context().get_world_rank() == 0


def summarize_model(model: nn.Module, inputs: tuple, batch_size: int, logdir: Path):
    model_logbook = {}
    model_stats = summary(
        model=model,
        input_size=(1, *inputs[1:]),
        depth=5,
        col_width=25,
        col_names=["kernel_size", "output_size", "num_params"],
        row_settings=["var_names"],
        verbose=0,
        mode='eval'
    )
    train_stats = summary(
        model=model,
        input_size=inputs,
        depth=5,
        col_width=25,
        col_names=["kernel_size", "output_size", "num_params"],
        row_settings=["var_names"],
        verbose=1,
        mode='train'
    )

    with (logdir / 'model.log').open('w') as f:
        f.write(str(model_stats))

    model_logbook['training_batch_size'] = batch_size
    model_logbook['input_bytes'] = model_stats.total_input
    model_logbook['total_params'] = model_stats.total_params
    model_logbook['trainable_params'] = model_stats.trainable_params
    model_logbook['param_bytes'] = model_stats.total_param_bytes

    model_logbook['eval_macs'] = model_stats.total_mult_adds
    model_logbook['training_macs'] = train_stats.total_mult_adds

    model_logbook['forward_pass_bytes'] = model_stats.total_output_bytes
    model_logbook['forward_backward_pass_bytes'] = train_stats.total_output_bytes

    model_logbook['eval_model_bytes'] = model_logbook['param_bytes'] + model_logbook['forward_pass_bytes']
    model_logbook['training_model_bytes'] = model_logbook['param_bytes'] + model_logbook['forward_backward_pass_bytes']

    model_logbook['eval_bytes'] = model_logbook['input_bytes'] + model_logbook['eval_model_bytes']
    model_logbook['training_bytes'] = model_logbook['input_bytes'] + model_logbook['training_model_bytes']

    model_logbook['layers'] = {}
    for layer in train_stats.summary_list:
        if layer.is_leaf_layer:
            model_logbook['layers'][f'{layer.class_name}_{layer.var_name}'] = {
                'macs': layer.macs,
                'params': max(layer.num_params, 0),
                'param_bytes': layer.param_bytes,
                'forward_pass_bytes': layer.output_bytes,
                'forward_backward_pass_bytes': layer.output_bytes * 2, # x2 for gradients
                'output_shape': layer.output_size,
            }

    with (logdir / 'model_logbook.json').open('w') as f:
        ujson.dump(
            model_logbook,
            f,
            indent=4,
            sort_keys=False,
            ensure_ascii=False,
            escape_forward_slashes=False
        )


def restore_model(config: DictConfig):
    outdir = Path(config.outdir)
    try: # check if model already exists
        checkpoints = [
            d for d in outdir.rglob('checkpoint*')
            if d.is_dir() and (
                (d / 'best_model' / 'zero_pp_rank_0_mp_rank_00_model_states.pt').exists() or
                (d / 'best_model.bin').exists()
            )
        ]
        checkpoints.sort(key=os.path.getctime)
        logger.info(f"Available checkpoints: {checkpoints}")

        logdir = Path(config.logdir)
        logger.info(f"{logdir / 'logbook.csv'}: {(logdir / 'logbook.csv').exists()}")
        training_history = pd.read_csv(logdir / 'logbook.csv', header=0, index_col=0).dropna(axis=0, how='any')
        logger.info(f"Training history\n{training_history}")

        latest_checkpoint = checkpoints[-1]
        starting_epoch = training_history.index.values[-1]

        overall_step = 0
        best_loss = training_history.loc[starting_epoch, 'loss']
        logger.info(f"Restoring from {latest_checkpoint} epoch {starting_epoch} with loss {best_loss}")

        starting_epoch += 1
        step_logbook = {}
        epoch_logbook = training_history.to_dict(orient='index')
        epoch_left = config.epochs - starting_epoch
        logger.info(epoch_logbook)

        logger.info(f"Epochs left {epoch_left}")
        restored = True

        if epoch_left == 0:
            return
        
    except Exception as e:
        restored = False
        latest_checkpoint = None
        logger.warning(e)
        logger.warning(f"No model found in {config.outdir}")
        best_loss, overall_step, starting_epoch = np.inf, 0, 0
        step_logbook, epoch_logbook = {}, {}

    return restored, latest_checkpoint, best_loss, overall_step, starting_epoch, step_logbook, epoch_logbook


def get_lr_scheduler(opt: torch.optim.Optimizer, steps_per_epoch: int, config: DictConfig, decay: str = 'cosine'):
    if config.fixedlr:
        scheduler = LinearLR(
            opt,
            start_factor=1.0,
            end_factor=1.0,
            total_iters=config.epochs,
        )
        logger.info(f"Training steps: [{steps_per_epoch * config.epochs}]")
    else:
        decay_epochs = config.epochs - (config.warmup + config.cooldown)
        total_steps = config.epochs * steps_per_epoch
        warmup_steps = config.warmup * steps_per_epoch
        cooldown_steps = config.cooldown * steps_per_epoch
        decay_steps = total_steps - (warmup_steps + cooldown_steps)

        logger.info(
            f"Training [epochs: {config.epochs} = total_steps: {total_steps}, "
            f"warmup: {config.warmup} = warmup_steps: {warmup_steps}, "
            f"cooldown: {config.cooldown} = cooldown_steps: {cooldown_steps}, "
            f"decay_epochs: {decay_epochs}," 
            f"decay_steps: {decay_steps}]"
        )

        scheduler, num_epochs = create_scheduler_v2(
            optimizer=opt,
            sched=decay,
            num_epochs=config.epochs,
            warmup_epochs=config.warmup,
            cooldown_epochs=config.cooldown,
            decay_epochs=decay_epochs,
            min_lr=1e-8,
            warmup_lr=1e-8,
        )

    return scheduler


def get_optimizer(params, config: DictConfig, optimizer: str, steps_per_epoch: int, deepspeed_scheduler: bool = False):
    if optimizer == 'adamw':
        opt = FusedAdam(
            params,
            lr=config.lr,
            weight_decay=config.wd,
            betas=(0.9, 0.99),
            eps=1e-08,
        )
    elif optimizer == 'lamb':
        opt = FusedLamb(
            params,
            lr=config.lr,
            weight_decay=config.wd,
            betas=(0.9, 0.99),
            eps=1e-08,
        )
    else:
        raise ValueError(f"Optimizer {optimizer} not supported")

    if deepspeed_scheduler:
        decay_epochs = config.epochs - (config.warmup + config.cooldown)
        total_steps = config.epochs * steps_per_epoch
        warmup_steps = config.warmup * steps_per_epoch
        decay_steps = total_steps - warmup_steps

        logger.info(
            f"Training [epochs: {config.epochs}, steps_per_epoch: {steps_per_epoch} = total_steps: {total_steps}, "
            f"warmup: {config.warmup} = warmup_steps: {warmup_steps}, decay_epochs: {decay_epochs},"
            f"decay_steps: {decay_steps}]"
        )

        scheduler = WarmupCosineLR(
            optimizer=opt,
            total_num_steps=total_steps,
            warmup_num_steps=warmup_steps,
            warmup_min_ratio=0.0,
            cos_min_ratio=0.0001,
            warmup_type='linear',
        )

        return opt, scheduler
    else:
        return opt


def _load_checkpoint(model_engine, opt, config, logger):
    checkpoint = get_checkpoint() 
    if checkpoint is not None:
        checkpointdir = checkpoint.as_directory()
        load_checkpoint(model_engine, opt, config, logger, checkpointdir)
    else:
        checkpointdir = Path(config.checkpointdir)
        load_checkpoint(model_engine, opt, config, logger, checkpointdir)


def supervised(config: DictConfig):
    restored, latest_checkpoint, best_loss, overall_step, starting_epoch, step_logbook, epoch_logbook = restore_model(config)
    
    if config.datasets.split:
        train_dataloader, val_dataloader = gather_dataset(config) 
    else: 
        train_dataloader = gather_dataset(config)
        val_dataloader = None
    
    steps_per_epoch = int(np.ceil(len(train_dataloader) / (config.gpu_workers * config.workers)))
    val_steps_per_epoch = int(np.ceil(len(val_dataloader) / (config.gpu_workers * config.workers))) if val_dataloader else None

    train_dataloader = raytorch.prepare_data_loader(train_dataloader)
    val_dataloader = raytorch.prepare_data_loader(val_dataloader) if val_dataloader else None

    # TODO: Better way to do this is probably with a registry as in: https://mmengine.readthedocs.io/en/latest/advanced_tutorials/registry.html
    #       But this works for now.
    model = build_dependency_graph_and_instantiate(config.models)
    
    # Skipping the model summary for now
    # input_shape = tuple(OmegaConf.to_container(config.datasets.inputs, resolve=True))
    # summarize_model(
    #     model=model,
    #     # inputs=config.inputs,
    #     inputs=([torch.ones(input_shape),None]),
    #     batch_size=config.datasets.batch_size,
    #     logdir=Path(config.logdir),
    # )

    opt = get_optimizer(
        params=model.parameters(),
        config=config,
        optimizer=config.opt,
        steps_per_epoch=steps_per_epoch
    )
    scheduler = get_lr_scheduler(
        opt=opt,
        config=config,
        steps_per_epoch=steps_per_epoch
    )

    model, opt, _, _ = initialize(
        model=model,
        optimizer=opt,
        config=OmegaConf.to_container(config.deepspeed_config, resolve=True)
    )

    if restored:
        _load_checkpoint(
            model_engine=model,
            opt=opt,
            config=config,
            logger=logger,
        )

    evaluator = instantiate(config.metrics)
    
    loss_nans = 0
    ray_context = get_context()
    with torch.autograd.set_detect_anomaly(True, check_nan=False):
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU],
            schedule=torch.profiler.schedule(skip_first=1, warmup=1, active=3, repeat=2, wait=1),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
            with_modules=True,
        ) if config.profile else nullcontext() as profiler:
            for epoch in range(starting_epoch, config.epochs):
                if ray_context.get_world_size() > 1:
                    train_dataloader.sampler.set_epoch(epoch)

                epoch_time = time.time()
                loss = 0.0
                step_times, step_utilization, step_vram = [], [], []

                for step, (inputs, targets) in enumerate(train_dataloader):
                    if step > 2:
                        break
                    step_time = time.time()
                    lr_value = opt.param_groups[0]["lr"]
                    # TODO: More modular loss computation design
                    loss_dict, outputs = model(inputs, targets)

                    # TODO: Make this logic more general
                    step_loss = sum(loss_dict.values())

                    if torch.isnan(step_loss):
                        loss_nans += 1
                        logger.warning(f"Step loss is {step_loss} for step {step} in epoch {epoch}")
                        if loss_nans > 5:
                            raise Exception(f"Step loss is {step_loss} for step {step} in epoch {epoch}")

                    model.backward(step_loss)
                    model.step()
                    scheduler.step(epoch)

                    cuda_util = torch.cuda.utilization()
                    cuda_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
                    loss += step_loss.detach().float()
                    overall_step += 1
                    step_timer = time.time() - step_time

                    step_times.append(step_timer)
                    step_utilization.append(cuda_util)
                    step_vram.append(cuda_vram)

                    step_log_entry = {
                        "step_loss": step_loss.detach().float().item(),
                        "step_lr": lr_value,
                        "step_timer": step_timer,
                        "cuda_vram": cuda_vram,
                        "step_utilization": cuda_util,
                    }

                    # record individual component losses
                    for name, val in loss_dict.items():
                        step_log_entry[name] = val.detach().float().item()

                    step_logbook[overall_step] = step_log_entry

                    if step % config.log_step == 0:
                        logger.info(f"Epoch {epoch} | Step {step}/{steps_per_epoch} | Loss: {step_loss.item():.4f}")

                mem_log = torch.cuda.memory_summary()
                logger.info(mem_log)
                with (Path(config.logdir) / 'memory.log').open('w') as f:
                    f.write(str(mem_log))

                loss = loss.item() / steps_per_epoch
                step_timer_avg = np.mean(step_times)
                epoch_timer = time.time() - epoch_time
                remaining_epochs = config.epochs - (epoch + 1)
                eta = epoch_timer * remaining_epochs / 3600
                cuda_utilization_avg = np.mean(step_utilization)
                cuda_memory_allocated_avg = np.mean(step_vram)
                max_cuda_memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)

                logger.info(f"│ training_epoch: {epoch+1}/{config.epochs}")
                logger.info(f"│ epoch_loss: {loss:.4g}")
                logger.info(f"│ epoch_lr: {lr_value:.4g}")
                logger.info(f"│ cuda_utilization: {cuda_utilization_avg:.0f}%")
                logger.info(f"│ cuda_memory_allocated: {cuda_memory_allocated_avg:.4g} GB")
                logger.info(f"│ max_cuda_memory_allocated: {max_cuda_memory_allocated:.4g} GB")
                logger.info(f"│ step_timer: {step_timer_avg * 1000:.0f}ms")
                logger.info(f"│ epoch_timer: {epoch_timer:.0f}s")
                logger.info(f"│ ETA: {eta:.2f}h")

                epoch_logbook[epoch] = {
                    "loss": loss,
                    "lr": lr_value,
                    "cuda_utilization": cuda_utilization_avg,
                    "cuda_memory_allocated": cuda_memory_allocated_avg,
                    "max_cuda_memory_allocated": max_cuda_memory_allocated,
                    "step_timer": step_timer_avg,
                    "epoch_timer": epoch_timer,
                    "eta": eta,
                }

                if val_dataloader is not None and (epoch + 1) % config.val_interval == 0:
                    
                    val_step_times = []
                    with inference_context(model):
                        with torch.no_grad():
                            for val_step, (inputs, targets) in enumerate(val_dataloader):
                                val_step_time = time.time()
                                loss_dict, outputs  = model(inputs, targets)
                                evaluator.process(targets, outputs)
                                val_step_times.append(time.time() - val_step_time)
                                if val_step % config.val_log_step == 0:
                                    metric_results, _ = evaluator.evaluate()
                                    logger.info(f"│ Epoch (val): {epoch+1}/{config.epochs}")
                                    logger.info(f"│ Step (val): {val_step}/{val_steps_per_epoch}")
                                    for k, v in metric_results.items():
                                        logger.info(f"│ val_{k}: {v}")

                    metric_results, ckpt_loss = evaluator.evaluate()
                    evaluator.reset()

                    val_step_timer_avg = np.mean(val_step_times)

                    logger.info(f"│ epoch (val): {epoch+1}/{config.epochs}")
                    logger.info(f"│ step_timer (val): {val_step_timer_avg * 1000:.0f}ms")
                    for k, v in metric_results.items():
                        logger.info(f"│ val_{k}: {v}")

                    epoch_logbook[epoch].update({
                        "val_step_timer": val_step_timer_avg,
                        **{f"val_{k}": v for k, v in metric_results.items()},
                    })

                    if ckpt_loss < best_loss:
                        best_loss = ckpt_loss

                        if config.deepspeed_config.zero_optimization.stage == 3:
                            model.save_checkpoint(config.checkpointdir, tag="best_model")
                        else:
                            torch.save(model.state_dict(), Path(config.checkpointdir) / "best_model.bin")
                            torch.save(opt.state_dict(), Path(config.checkpointdir) / "best_optimizer.bin")

                pd.DataFrame.from_dict(epoch_logbook, orient='index').to_csv(Path(config.logdir) / 'logbook.csv')
                pd.DataFrame.from_dict(step_logbook, orient='index').to_csv(Path(config.logdir) / 'steplogbook.csv')
                
                checkpoint = Checkpoint.from_directory(config.checkpointdir)
                report(metrics=epoch_logbook[epoch], checkpoint=checkpoint)

                if is_main_process():
                    logger.info(epoch_logbook[epoch])

                if config.profile:
                    profiler.step()

        if config.deepspeed_config.zero_optimization.stage == 3:
            model.save_checkpoint(config.checkpointdir, tag="last_model")
        else:
            # weights on each rank are placeholders in deepspeed stage 3
            # and hence cannot be saved directly with torch.save
            # instead convert with convert_zero_checkpoint (see checkpoint.py)
            torch.save(model.state_dict(), Path(config.checkpointdir) / "last_model.bin")
            torch.save(opt.state_dict(), Path(config.checkpointdir) / "last_optimizer.bin")

        checkpoint = Checkpoint.from_directory(config.checkpointdir)
        report(metrics=epoch_logbook[epoch], checkpoint=checkpoint)
