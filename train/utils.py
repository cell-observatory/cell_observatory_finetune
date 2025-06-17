import os
import sys
import ujson
import logging
from pathlib import Path
from typing import Dict, Union

from omegaconf import DictConfig

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torchinfo import summary
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LinearLR
from timm.scheduler import create_scheduler_v2

from deepspeed.ops.adam import FusedAdam
from deepspeed.ops.lamb import FusedLamb
from deepspeed.runtime.lr_schedules import WarmupCosineLR


logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_lr_scheduler(opt: torch.optim.Optimizer, 
                     steps_per_epoch: int, 
                     config: DictConfig, 
                     decay: str = 'cosine'
):
    if config.schedulers.fixedlr:
        scheduler = LinearLR(
            opt,
            start_factor=1.0,
            end_factor=1.0,
            total_iters=config.schedulers.epochs,
        )
        logger.info(f"Training steps: [{steps_per_epoch * config.schedulers.epochs}]")
    else:
        decay_epochs = config.schedulers.epochs - (config.schedulers.warmup + config.schedulers.cooldown)
        total_steps = config.schedulers.epochs * steps_per_epoch
        warmup_steps = config.schedulers.warmup * steps_per_epoch
        cooldown_steps = config.schedulers.cooldown * steps_per_epoch
        decay_steps = total_steps - (warmup_steps + cooldown_steps)

        logger.info(
            f"Training [epochs: {config.schedulers.epochs} = total_steps: {total_steps}, "
            f"warmup: {config.schedulers.warmup} = warmup_steps: {warmup_steps}, "
            f"cooldown: {config.schedulers.cooldown} = cooldown_steps: {cooldown_steps}, "
            f"decay_epochs: {decay_epochs}," 
            f"decay_steps: {decay_steps}]"
        )

        scheduler, num_epochs = create_scheduler_v2(
            optimizer=opt,
            sched=decay,
            num_epochs=config.schedulers.epochs,
            warmup_epochs=config.schedulers.warmup,
            cooldown_epochs=config.schedulers.cooldown,
            decay_epochs=decay_epochs,
            min_lr=1e-8,
            warmup_lr=1e-8,
        )

    return scheduler


def get_optimizer(params, 
                  config: DictConfig, 
                  optimizer: str, 
                  steps_per_epoch: int, 
                  deepspeed_scheduler: bool = False
):
    if optimizer == 'adamw':
        opt = FusedAdam(
            params,
            lr=config.optimizers.lr,
            weight_decay=config.optimizers.wd,
            betas=(0.9, 0.99),
            eps=1e-08,
        )
    elif optimizer == 'lamb':
        opt = FusedLamb(
            params,
            lr=config.optimizers.lr,
            weight_decay=config.optimizers.wd,
            betas=(0.9, 0.99),
            eps=1e-08,
        )
    else:
        raise ValueError(f"Optimizer {optimizer} not supported")

    if deepspeed_scheduler:
        decay_epochs = config.schedulers.epochs - (config.schedulers.warmup + config.schedulers.cooldown)
        total_steps = config.schedulers.epochs * steps_per_epoch
        warmup_steps = config.schedulers.warmup * steps_per_epoch
        decay_steps = total_steps - warmup_steps

        logger.info(
            f"Training [epochs: {config.schedulers.epochs}, steps_per_epoch: \
                {steps_per_epoch} = total_steps: {total_steps}, "
            f"warmup: {config.schedulers.warmup} = warmup_steps: \
                {warmup_steps}, decay_epochs: {decay_epochs},"
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
        return opt, None


def get_steps_per_epoch(train_dataloader, val_dataloader, config: DictConfig):
    steps_per_epoch = int(np.ceil(len(train_dataloader) / (config.clusters.total_gpus)))
    val_steps_per_epoch = int(np.ceil(len(val_dataloader) / (config.clusters.total_gpus))) \
        if val_dataloader else None
    return steps_per_epoch, val_steps_per_epoch


# TODO: store best loss, starting epoch and starting step
#       using save method in checkpoint manager inside client state
#       this way no matter what checkpoint we decide to load
#       we always have the best loss, starting epoch and starting step
#       to resume training from directly in the checkpoint
#       resume model state is most useful when restarting a 
#       job from an earlier checkpoint to sidestep training 
#       instabilities hence its useful to be able to restart
#       from any checkpoint where we only have to specify
#       the checkpoint directory and the checkpoint tag
#       see: https://arxiv.org/pdf/2204.02311
def resume_model_state(config: DictConfig, checkpoint_manager):
    assert config.checkpoint.load_checkpointdir is not None and \
        Path(config.checkpoint.load_checkpointdir).is_dir(), \
        f"Checkpoint directory does not exist: {config.checkpoint.load_checkpointdir}" \
        f"Checkpoint directory must be populated " \
        f"with a valid checkpoint to resume training."
    
    # load checkpoint if resuming a job or loading a checkpoint
    # from save location 
    ckpt_path, client_state = checkpoint_manager.load(
        prefix=config.checkpoint.checkpoint_tag,
        dtype=config.quantization,
        engine=config.engine,
        num_ckpt_shards=config.checkpoint.num_ckpt_shards
    )

    # get best loss and starting epoch from client state saved 
    # in checkpoint_manager save method
    best_loss, starting_epoch = client_state["best_loss"], client_state["epoch"]

    # get epochs left     
    epochs_left = config.schedulers.epochs - starting_epoch

    # TODO: store in client state as well
    # get start step
    logdir = Path(config.logging.logdir)
    if not (logdir / 'scalars' / 'step_logbook.csv').exists():
        raise FileNotFoundError(f"Log book not found in {logdir}")
    training_history = pd.read_csv(logdir / 'logbook.csv', header=0, index_col=0)    
    training_history = training_history[training_history['epoch'] <= starting_epoch]
    starting_step = training_history['iter'].max()

    if epochs_left <= 0:
        raise ValueError(
            f"No epochs left to train. Starting epoch {starting_epoch} "
            f"exceeds total epochs {config.schedulers.epochs}."
        )

    return best_loss, starting_step, starting_epoch


def resume_run(trainer, config: DictConfig):
    # NOTE: load_checkpointdir and checkpoint_manager.checkpointdir
    # are two different paths, the former is used to load
    # a pre-existing checkpoint, the latter is used to save
    # new checkpoints for the current run
    if config.checkpoint.resume_run:
        best_loss, iter, epoch = resume_model_state(config, checkpoint_manager=trainer.checkpoint_manager)
        
        trainer.event_recorder.resume(
            iter=iter, 
            epoch=epoch
        )

    else:
        epoch, iter, best_loss = 0, 0, np.inf

        Path(config.outdir).mkdir(exist_ok=True, parents=True)
        Path(config.logging.logdir).mkdir(exist_ok=True, parents=True)
        Path(config.checkpoint.checkpoint_manager.checkpointdir).mkdir(exist_ok=True, parents=True)

        logger.info(f"Output dir: {config.outdir}")
        logger.info(f"Log dir: {config.logging.logdir}")
        logger.info(f"Checkpoint save dir: {config.checkpoint.checkpoint_manager.checkpointdir}")
    
    return best_loss, iter, epoch


def summarize_model(model: nn.Module, inputs: tuple, batch_size: int, logdir: Path):
    model_logbook = {}
    model_stats = summary(
        model=model,
        input_size=inputs,
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

    model_logbook['eval_model_bytes'] = model_logbook['param_bytes'] \
        + model_logbook['forward_pass_bytes']
    model_logbook['training_model_bytes'] = model_logbook['param_bytes'] \
        + model_logbook['forward_backward_pass_bytes']

    model_logbook['eval_bytes'] = model_logbook['input_bytes'] + \
        model_logbook['eval_model_bytes']
    model_logbook['training_bytes'] = model_logbook['input_bytes'] + \
        model_logbook['training_model_bytes']

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