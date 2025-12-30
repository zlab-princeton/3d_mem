import os
import time
import wandb
import numpy as np
from PIL import Image
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from typing import Tuple, Generic, Dict, Callable, Optional, Any
from pprint import pprint

import torch
import torchvision
import pytorch_lightning as pl
import pytorch_lightning.loggers
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers.logger import DummyLogger
from pytorch_lightning.utilities import rank_zero_only, rank_zero_info
from pytorch_lightning.callbacks import Callback

from functools import wraps

def node_zero_only(fn: Callable) -> Callable:
    @wraps(fn)
    def wrapped_fn(*args, **kwargs) -> Optional[Any]:
        if node_zero_only.node == 0:
            return fn(*args, **kwargs)
        return None
    return wrapped_fn

node_zero_only.node = getattr(node_zero_only, 'node', int(os.environ.get('NODE_RANK', 0)))

def node_zero_experiment(fn: Callable) -> Callable:
    """Returns the real experiment on rank 0 and otherwise the DummyExperiment."""
    @wraps(fn)
    def experiment(self):
        @node_zero_only
        def get_experiment():
            return fn(self)
        return get_experiment() or DummyLogger.experiment
    return experiment

# customize wandb for node 0 only
class MyWandbLogger(WandbLogger):
    @WandbLogger.experiment.getter
    @node_zero_experiment
    def experiment(self):
        return super().experiment

class SetupCallback(Callback):
    def __init__(self, config: DictConfig, exp_config: DictConfig,
                 basedir: Path, logdir: str = "log", ckptdir: str = "ckpt") -> None:
        super().__init__()
        self.logdir = basedir / logdir
        self.ckptdir = basedir / ckptdir
        self.config = config
        self.exp_config = exp_config

    # def on_pretrain_routine_start(self, trainer: pl.trainer.Trainer, pl_module: pl.LightningModule) -> None:
    #     if trainer.global_rank == 0:
    #         # Create logdirs and save configs
    #         os.makedirs(self.logdir, exist_ok=True)
    #         os.makedirs(self.ckptdir, exist_ok=True)
    #
    #         print("Experiment config")
    #         print(self.exp_config.pretty())
    #
    #         print("Model config")
    #         print(self.config.pretty())

    def on_fit_start(self, trainer: pl.trainer.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)

            # print("Experiment config")
            # pprint(self.exp_config)
            #
            # print("Model config")
            # pprint(self.config)



class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.root_gpu)
        torch.cuda.synchronize(trainer.root_gpu)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        torch.cuda.synchronize(trainer.root_gpu)
        max_memory = torch.cuda.max_memory_allocated(trainer.root_gpu) / 2 ** 20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass


