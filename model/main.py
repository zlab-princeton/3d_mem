import os
import re
import argparse
import warnings
from pathlib import Path
from typing import Tuple, List, Optional, Dict

import torch
import torch._dynamo
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.strategies import DDPStrategy, DeepSpeedStrategy
from pytorch_lightning.loggers import Logger, TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_info
from omegaconf import OmegaConf, DictConfig
from einops._torch_specific import allow_ops_in_compiled_graph

from hy3dshape.utils import get_config_from_file, instantiate_from_config
from hy3dshape.utils.trainings.force_resume_counters import ForceResumeCounters

warnings.filterwarnings("ignore", category=UserWarning, module="torch._dynamo")
torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.force_parameter_static_shapes = False 
allow_ops_in_compiled_graph()


def infer_resume_step_from_ckpt_path(path: Optional[str]) -> Optional[int]:
    """
    Infers the training step from a checkpoint path.
    Supported patterns:
      - ckpt-step=XXXXX(.ckpt)
      - checkpoint-XXXK / checkpoint-XXXk
    """
    if not path:
        return None

    parts = os.path.normpath(path).split(os.sep)[::-1]
    pat_ckpt_step = re.compile(r'ckpt-step=(\d+)')
    pat_checkpoint_k = re.compile(r'checkpoint-(\d+)[kK]\b')

    for comp in parts:
        # ckpt-step=XXXXX(.ckpt)
        m = pat_ckpt_step.search(comp)
        if m:
            return int(m.group(1))

        # checkpoint-XXXK / checkpoint-XXXk
        m = pat_checkpoint_k.search(comp)
        if m:
            return int(m.group(1)) * 1000

    return None


class SetupCallback(Callback):
    def __init__(self, config: DictConfig, basedir: Path, logdir: str = "log", ckptdir: str = "ckpt") -> None:
        super().__init__()
        self.logdir = basedir / logdir
        self.ckptdir = basedir / ckptdir
        self.config = config

    def on_fit_start(self, trainer: pl.trainer.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.global_rank == 0:
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)


class DynamoResetCallback(pl.Callback):
    """Resets Torch Dynamo cache periodically to prevent OOM/slowdowns during long training."""
    def __init__(self, every_n_val: int = 0):
        self.every_n_val = every_n_val
        self._n = 0

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if not getattr(pl_module, "torch_compile", False):
            return
        if self.every_n_val <= 0:
            return
        self._n += 1
        if self._n % self.every_n_val != 0:
            return
        rank_zero_info("[DynamoReset] torch._dynamo.reset() after validation")
        torch._dynamo.reset()


def setup_callbacks(config: DictConfig) -> Tuple[List[Callback], Logger]:
    training_cfg = config.training
    
    out_dir = training_cfg.get("output_dir", "outputs")
    basedir = Path(out_dir)
    os.makedirs(basedir, exist_ok=True)
    
    all_callbacks = []

    setup_callback = SetupCallback(config, basedir)
    all_callbacks.append(setup_callback)
    
    all_callbacks.append(DynamoResetCallback(every_n_val=20))
    
    # Force resume counters if init_ckpt is provided but we aren't doing a full PL restore
    if bool(training_cfg.get("init_ckpt")) and not bool(training_cfg.get("ckpt_path")):
        all_callbacks.append(ForceResumeCounters(lambda m: getattr(m, "resume_step", 0)))
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=setup_callback.ckptdir,
        filename="ckpt-{step:08d}",
        monitor=training_cfg.monitor,
        mode="max",
        save_top_k=-1, # Save all checkpoints specified by every_n_train_steps
        verbose=False,
        every_n_train_steps=training_cfg.every_n_train_steps
    )
    all_callbacks.append(checkpoint_callback)

    if "callbacks" in config:
        for key, value in config['callbacks'].items():
            custom_callback = instantiate_from_config(value)
            all_callbacks.append(custom_callback)

    logger = TensorBoardLogger(
        save_dir=str(setup_callback.logdir),
        name="tensorboard",
        default_hp_metric=False,
        log_graph=False,
        max_queue=10,
        flush_secs=30 
    )
    return all_callbacks, logger


def merge_cfg(cfg, arg_cfg):
    """Merges command line arguments into the config.training dictionary."""
    for key in arg_cfg.keys():
        if arg_cfg[key] is not None:
            if key in cfg.training or key == "output_dir":
                cfg.training[key] = arg_cfg[key]
    cfg.training = DictConfig(cfg.training)
    return cfg


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action='store_true', help="Enable TF32 and other speedups")
    
    # config & environment
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-nn", "--num_nodes", type=int, default=1)
    parser.add_argument("-ng", "--num_gpus", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")

    # training hyperparams
    parser.add_argument("-u", "--update_every", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("-st", "--steps", type=int, default=50000000)
    parser.add_argument("-lr", "--base_lr", type=float, default=4.5e-6)
    parser.add_argument("-a", "--use_amp", default=False, action="store_true")
    parser.add_argument("--amp_type", type=str, default="16", choices=["16", "bf16", "32"])
    
    # checkpointing & logging
    parser.add_argument("--gradient_clip_val", type=float, default=None)
    parser.add_argument("--gradient_clip_algorithm", type=str, default="norm")
    parser.add_argument("--every_n_train_steps", type=int, default=50000)
    parser.add_argument("--log_every_n_steps", type=int, default=50)
    parser.add_argument("--val_check_interval", type=int, default=1024)
    parser.add_argument("--limit_val_batches", type=int, default=64)
    parser.add_argument("--monitor", type=str, default="val/total_loss")
    
    # resuming / strategies
    parser.add_argument("--ckpt_path", type=str, default="", help="Path to resume full training state")
    parser.add_argument("--init_ckpt", type=str, default="", help="Path to load weights only")
    parser.add_argument("--deepspeed", default=False, action="store_true", help="Use DeepSpeed Stage 1")
    parser.add_argument("--deepspeed2", default=False, action="store_true", help="Use DeepSpeed Stage 2")
    parser.add_argument("--scale_lr", type=bool, nargs="?", const=True, default=False,
                        help="Auto-scale base-lr by ngpu * batch_size * n_accumulate")
    
    return parser.parse_args()
    

if __name__ == "__main__":
    
    args = get_args()
    
    if args.fast:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision('medium')
        torch.utils.data._utils.MP_STATUS_CHECK_INTERVAL = 0.05

    pl.seed_everything(args.seed, workers=True)

    config = get_config_from_file(args.config)
    config = merge_cfg(config, vars(args))
    training_cfg = config.training

    rank_zero_info("Begin to print configuration ...")
    rank_zero_info(OmegaConf.to_yaml(config))
    rank_zero_info("Finish print ...")

    callbacks, loggers = setup_callbacks(config)
    data: pl.LightningDataModule = instantiate_from_config(config.dataset)
    model: pl.LightningModule = instantiate_from_config(config.model)

    if training_cfg.get("init_ckpt"):
        rank_zero_info(f"Loading weights from {training_cfg.init_ckpt}")
        model.load_weights_only(training_cfg.init_ckpt)
        step = infer_resume_step_from_ckpt_path(training_cfg.init_ckpt)
        if step is not None:
            model.resume_step = step
    
    nodes = args.num_nodes
    ngpus = args.num_gpus
    base_lr = training_cfg.base_lr
    accumulate_grad_batches = training_cfg.update_every
    batch_size = config.dataset.params.batch_size

    if 'SLURM_NNODES' in os.environ:
        nodes = int(os.environ['SLURM_NNODES'])
    elif 'NNODES' in os.environ:
        nodes = int(os.environ['NNODES'])

    training_cfg.num_nodes = nodes
    args.num_nodes = nodes

    if args.scale_lr:
        model.learning_rate = accumulate_grad_batches * nodes * ngpus * batch_size * base_lr
        info = f"Setting learning rate to {model.learning_rate:.2e} = {accumulate_grad_batches} (accumulate)"
        info += f" * {nodes} (nodes) * {ngpus} (num_gpus) * {batch_size} (batchsize) * {base_lr:.2e} (base_lr)"
        rank_zero_info(info)
    else:
        model.learning_rate = base_lr
        rank_zero_info("++++ NOT USING LR SCALING ++++")
        rank_zero_info(f"Setting learning rate to {model.learning_rate:.2e}")

    strategy = None
    if args.num_nodes > 1 or args.num_gpus > 1:
        if args.deepspeed:
            strategy = DeepSpeedStrategy(stage=1)
        elif args.deepspeed2:
            strategy = 'deepspeed_stage_2'
        else:
            # Standard DDP
            strategy = DDPStrategy(find_unused_parameters=False, bucket_cap_mb=400, static_graph=True)

    rank_zero_info(f'*' * 100)
    if training_cfg.use_amp:
        amp_type = training_cfg.amp_type
        assert amp_type in ['bf16', '16', '32'], f"Invalid amp_type: {amp_type}"
        rank_zero_info(f'Using {amp_type} precision')
    else:
        amp_type = "32-true"
        rank_zero_info(f'Using 32 bit precision')
    rank_zero_info(f'*' * 100)

    trainer = pl.Trainer(
        max_steps=training_cfg.steps,
        precision=amp_type,
        callbacks=callbacks,
        accelerator="gpu",
        devices=args.num_gpus,
        num_nodes=training_cfg.num_nodes,
        gradient_clip_val=training_cfg.get('gradient_clip_val'),
        gradient_clip_algorithm=training_cfg.get('gradient_clip_algorithm'),
        accumulate_grad_batches=args.update_every,
        strategy=strategy,
        logger=loggers,
        log_every_n_steps=training_cfg.log_every_n_steps,
        val_check_interval=training_cfg.val_check_interval,
        limit_val_batches=training_cfg.limit_val_batches,
        check_val_every_n_epoch=None,
        num_sanity_val_steps=0
    )

    ckpt_path = training_cfg.ckpt_path if training_cfg.ckpt_path else None
    trainer.fit(model, datamodule=data, ckpt_path=ckpt_path)