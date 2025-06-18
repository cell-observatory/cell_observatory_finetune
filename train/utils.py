import os
import sys
import math
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
#       with checkpoint manager in client state
#       resume model state is most useful when restarting a 
#       job from an earlier checkpoint to sidestep training 
#       instabilities with resume_model_state only the checkpoint
#       directory and the checkpoint tag need be specified whereafter
#       any checkpoint with corresponding iter, epoch, best_loss
#       will be loaded from the checkpoint directory.
#       see: https://arxiv.org/pdf/2204.02311 for
#       strategies to resume training after training 
#       instabilities
def resume_model_state(config: DictConfig, checkpoint_manager):
    assert config.checkpoint.checkpoint_manager.resume_checkpointdir is not None and \
        Path(config.checkpoint.checkpoint_manager.resume_checkpointdir).is_dir(), \
        f"Checkpoint directory does not exist: {config.checkpoint.checkpoint_manager.resume_checkpointdir}" \
        f"Checkpoint directory must be populated " \
        f"with a valid checkpoint to resume training."
    
    ckpt_path, client_state = checkpoint_manager.load()

    # get metadata from client state
    best_loss = client_state["best_loss"]
    starting_epoch, starting_iter = client_state["epoch"], client_state["iter"]
    epochs_left = config.schedulers.epochs - starting_epoch

    if epochs_left <= 0:
        raise ValueError(
            f"No epochs left to train. Starting epoch {starting_epoch} "
            f"exceeds total epochs {config.schedulers.epochs}."
        )
    
    if config.checkpoint.checkpoint_manager.resume_checkpointdir != \
        config.checkpoint.checkpoint_manager.save_checkpointdir:
        logger.warning(
            f"Checkpoint resume directory {config.checkpoint.checkpoint_manager.resume_checkpointdir} "
            f"does not match new save checkpoint directory {config.checkpoint.checkpoint_manager.save_checkpointdir}. "
            "New checkpoints will NOT be saved to the previous checkpoint directory!"
        )
        Path(config.checkpoint.checkpoint_manager.save_checkpointdir).mkdir(exist_ok=True, parents=True)
    
    if not Path(config.logging.logdir).exists():
        logger.warning(
            f"Log directory {config.logging.logdir} does not exist. "
            f"Creating new log directory. New logs from starting epoch {starting_epoch} "
            f"will not contain any previous training run data!"
        )
        Path(config.logging.logdir).mkdir(exist_ok=True, parents=True) 

    return best_loss, starting_iter, starting_epoch


def resume_run(trainer, config: DictConfig):
    Path(config.outdir).mkdir(exist_ok=True, parents=True)
    if config.checkpoint.resume_run:
        best_loss, iter, epoch = resume_model_state(config, 
                                    checkpoint_manager=trainer.checkpoint_manager)        
        trainer.event_recorder.resume(
            iter=iter, 
            epoch=epoch
        )

    else:
        assert config.checkpoint.checkpoint_manager.save_checkpointdir is None, \
            "Checkpoint directory must be None when starting a new training run."
        assert config.logging.logdir is None, \
            "Checkpoint directory must be None when starting a new training run."
        
        epoch, iter, best_loss = 0, 0, np.inf

        Path(config.logging.logdir).mkdir(exist_ok=True, parents=True)
        Path(config.checkpoint.checkpoint_manager.save_checkpointdir).mkdir(exist_ok=True, parents=True)

        logger.info(f"Output dir: {config.outdir}")
        logger.info(f"Log dir: {config.logging.logdir}")
        logger.info(f"Checkpoint save dir: {config.checkpoint.checkpoint_manager.save_checkpointdir}")
    
    return best_loss, iter, epoch