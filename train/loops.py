"""
https://github.com/facebookresearch/detectron2/blob/main/detectron2/engine/train_loop.py
https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py

Apache License
Version 2.0, January 2004
http://www.apache.org/licenses/

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


import logging
import weakref
from pathlib import Path
from typing import List, Optional, Sequence

from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate, get_class

from ray.train import get_context

import torch
from deepspeed import initialize

from cell_observatory_finetune.train.utils import (
    get_optimizer,
    get_lr_scheduler,
    get_steps_per_epoch,
    resume_run
)
from cell_observatory_finetune.train.hooks import HookBase
from cell_observatory_finetune.utils.logging import EventRecorder
from cell_observatory_finetune.utils.comm import inference_context
from cell_observatory_finetune.data.dataloaders import get_dataloader
from cell_observatory_finetune.train.registry import build_dependency_graph_and_instantiate


logger = logging.getLogger("ray")
logger.setLevel(logging.INFO) # OR: DEBUG, WARNING, ERROR, CRITICAL

# silence broken logging call in Ray internals to prevent
# checkpoint saving from failing
logging.getLogger("ray.train._internal.checkpoint_manager").setLevel(logging.INFO)


# Ray train wrapper entry point
def train_loop_per_worker(config):
    trainer_cls = get_class(config.trainer)
    trainer_per_worker = trainer_cls(config)
    trainer_per_worker.run()
    return {"best_metric": trainer_per_worker.best_metric}


class BaseTrainer:
    """
    Base class for iterative trainer with hooks.
    """

    def __init__(self, config: DictConfig) -> None:
        # initialize event recorder
        self.event_recorder: EventRecorder = instantiate(config.logging.event_recorder)
        # initialize visualizer
        self.visualizer = instantiate(config.visualization)

        # initialize event_writers
        event_writers = self._build_event_writers(
            config.logging.event_writers, self.event_recorder, self.visualizer
        )
        self.event_writers_list = instantiate(
            config.logging.event_writers_list,
            writers = event_writers
        )
        
        # intialize hooks
        hooks = self._build_hooks(config.hooks.hooks_list, self.event_writers_list)
        self._hooks: List[HookBase] = []
        self.register_hooks(hooks)

    @staticmethod
    def _build_event_writers(w_cfgs, recorder, visualizer):
        writers = []
        for writer in w_cfgs:
            # TODO: is there a better way to do this?
            if writer._target_ == "cell_observatory_finetune.utils.logging.LocalEventWriter":
                writer = instantiate(
                    writer,
                    event_recorder=recorder,
                    visualizer=visualizer,
                )
            else:
                writer = instantiate(writer, event_recorder=recorder)
            writers.append(writer)
        return writers

    @staticmethod
    def _build_hooks(h_cfgs, event_writers):
        hooks = []
        for hc in h_cfgs:
            # inject writers into PeriodicWriter-like hooks
            # TODO: is there a better way to do this?
            if hc._target_ == "cell_observatory_finetune.train.hooks.PeriodicWriter":
                hook = instantiate(hc, writers=event_writers)
            else:
                hook = instantiate(hc)
            hooks.append(hook)
        return hooks

    def register_hooks(self, hooks: List[Optional[HookBase]]) -> None:
        """
        Register hooks to the trainer. The hooks are executed in the order
        they are registered.

        Args:
            hooks (list[Optional[HookBase]]): list of hooks
        """
        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            assert isinstance(h, HookBase)
            # to avoid circular reference, hooks and trainer cannot own each other
            # this normally does not matter, but will cause memory leak if the
            # involved objects contain __del__
            # hence we use weakref.proxy
            h.trainer = weakref.proxy(self)
        
        # reorder hooks by priority
        # higher priority hooks are executed first
        hooks.sort(key=lambda h: -h.PRIORITY.value)
        self._hooks.extend(hooks)

    def before_train(self):
        for h in self._hooks:
            h.before_train()

    def after_train(self):
        self.event_recorder._iter = self._iter
        self.event_recorder._epoch = self._epoch
        for h in self._hooks:
            h.after_train()

    def before_step(self):
        # maintain the invariant that 
        # event_recorder.iter == trainer.iter
        # for the entire execution of each step
        self.event_recorder._iter = self._iter

        for h in self._hooks:
            h.before_step()

    def after_backward(self):
        for h in self._hooks:
            h.after_backward()

    def after_step(self,*args, **kwargs):
        for h in self._hooks:
            h.after_step(*args, **kwargs)

    def before_epoch(self):
        # maintain the invariant that 
        # event_recorder.epoch == trainer.epoch
        # for the entire execution of each step
        self.event_recorder._epoch = self._epoch
        for h in self._hooks:
            h.before_epoch()
    
    def after_epoch(self, *args, **kwargs):
        for h in self._hooks:
            h.after_epoch(*args, **kwargs)

    def before_validation(self):
        self.event_recorder._val_iter = 0
        for h in self._hooks:
            h.before_validation()

    def after_validation(self):
        for h in self._hooks:
            h.after_validation()
    
    def before_val_step(self):
        # maintain the invariant that 
        # event_recorder.val_iter == trainer.val_iter
        # for the entire execution of each validation step
        self.event_recorder._val_iter = self._val_iter

        for h in self._hooks:
            h.before_val_step()

    def after_val_step(self, *args, **kwargs):
        for h in self._hooks:
            h.after_val_step(*args, **kwargs)

    def state_dict(self):
        ret = {"iteration": self.iter, "epoch": self._epoch}
        hooks_state = {}
        for h in self._hooks:
            sd = h.state_dict()
            if sd:
                name = type(h).__qualname__
                if name in hooks_state:
                    # TODO handle repetitive stateful hooks
                    continue
                hooks_state[name] = sd
        if hooks_state:
            ret["hooks"] = hooks_state
        return ret

    def load_state_dict(self, state_dict):
        self._iter = state_dict["iteration"]
        self._epoch = state_dict["epoch"]
        for key, value in state_dict.get("hooks", {}).items():
            for h in self._hooks:
                try:
                    name = type(h).__qualname__
                except AttributeError:
                    continue
                if name == key:
                    h.load_state_dict(value)
                    break
            else:
                logger.warning(f"Cannot find the hook '{key}', its state_dict is ignored.")


# most of the classes could probably be 
# instantiated with Hydra less explicitly
# but keeping here for clarity for now
class EpochBasedTrainer(BaseTrainer):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        self.ray_context = get_context()
        self.val_begin, self.val_interval = cfg.evaluation.val_begin, cfg.evaluation.val_interval
        self.stop_training, self._max_epochs = False, cfg.schedulers.epochs

        # initialize dataset and dataloader
        self.train_dataloader, self.val_dataloader = get_dataloader(cfg) 

        steps_per_epoch, val_steps_per_epoch = get_steps_per_epoch(
            train_dataloader=self.train_dataloader,
            val_dataloader=self.val_dataloader,
            config=cfg
        )

        # initialize model
        # TODO: consider migrating to BUILD() based initialization
        #       instead of recursive instantiation
        model = build_dependency_graph_and_instantiate(cfg.models)

        # initialize optimizer and learning rate scheduler
        opt, _ = get_optimizer(
            params=model.parameters(),
            config=cfg,
            optimizer=cfg.optimizers.opt,
            steps_per_epoch=steps_per_epoch
        )
        self.scheduler = get_lr_scheduler(
            opt=opt,
            config=cfg,
            steps_per_epoch=steps_per_epoch
        )

        # initialize deepspeed
        self.model, self.opt, _, _ = initialize(
            model=model,
            optimizer=opt,
            config=OmegaConf.to_container(cfg.deepspeed, resolve=True)
        )

        # initialize checkpoint manager
        self.checkpoint_manager = instantiate(
            cfg.checkpoint.checkpoint_manager,
            model=self.model
        )

        # if resume job, gather the state from the checkpoint
        # else intialize outdir, logdir, and checkpointdir
        # directories must be empty if not resuming a job
        # to avoid overwriting existing checkpoints
        # see finetune/train/utils.py:resume_run()
        best_metric, step, epoch = resume_run(self, cfg)
        self.start_epoch, self.start_iter, self.best_metric = epoch, step, best_metric
        self._epoch, self._iter, self._val_iter = self.start_epoch, self.start_iter, 0

        # initialize evaluator
        self.evaluator = instantiate(cfg.evaluation.evaluator)
    
    def run(self):
        """
        Launch training.
        """
        self.before_train()

        while self._epoch < self._max_epochs and not self.stop_training:
            self.run_epoch()

        self.after_train()

    def run_epoch(self) -> None:
        """
        Iterate one epoch.
        """
        self.before_epoch()
        for idx, data_sample in enumerate(self.train_dataloader):
            if idx > 200:
                break  # for testing purposes, remove later
            self.run_step(idx, data_sample)

        if self.val_dataloader and \
           (self._epoch >= self.val_begin and
            (self._epoch - self.val_begin) % self.val_interval == 0):
            # run validation
            self.run_validation()

        self.after_epoch()
        self._epoch += 1

    def run_step(self, idx, data_sample: Sequence[dict]) -> None:
        """
        Iterate one mini-batch.
        """
        self.before_step()
        
        loss_dict, outputs = self.model(data_sample)

        self.model.backward(loss_dict["step_loss"])
        self.model.step()

        self.event_recorder.put_scalars(
            scope="step",
            **{k: (v.item() if torch.is_tensor(v) else v)
            for k, v in loss_dict.items()
            }
        )

        self.after_step(data_sample=data_sample,
                        outputs=outputs, loss_dict=loss_dict)
        self._iter += 1

    def run_validation(self) -> None:
        """
        Run validation.
        """
        self.before_validation()
        # technically, contexts could probably be a hook
        # also but feels clearer to have them here
        with inference_context(self.model):
            with torch.no_grad():
                for idx, data_sample in enumerate(self.val_dataloader):
                    self.run_validation_step(idx, data_sample)
                    if idx > 100:
                        break  # for testing purposes, remove later

        metrics = self.evaluator.evaluate()
        self.event_recorder.put_scalars(
            scope="epoch",
            prefix="val_",
            **{k: (v.item() if torch.is_tensor(v) else v)
                for k, v in metrics.items()
            }
        )
        self.evaluator.reset()

        self.after_validation()
    
    def run_validation_step(self, idx: int, data_sample: Sequence[dict]) -> None:
        """
        Iterate one validation step.
        """
        self.before_val_step()

        loss_dict, outputs = self.model(data_sample)
        self.evaluator.process(data_sample, outputs, loss_dict)

        self.after_val_step(data_sample=data_sample, 
                            outputs=outputs, loss_dict=loss_dict)
        self._val_iter += 1