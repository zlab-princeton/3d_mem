import os
from contextlib import contextmanager, nullcontext
from typing import List, Tuple, Optional, Union
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch._dynamo as dynamo
from torch.optim import lr_scheduler
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
import torchvision
from torchvision.transforms.functional import to_tensor

from ...utils.ema import LitEma, DSEma
from ...utils.misc import instantiate_from_config, instantiate_non_trainable_model
import time

@contextmanager
def nvtx(name):
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_pop()

class Diffuser(pl.LightningModule):
    def __init__(
        self,
        *,
        first_stage_config,
        cond_stage_config,
        denoiser_cfg,
        scheduler_cfg,
        optimizer_cfg,
        pipeline_cfg=None,
        image_processor_cfg=None,
        lora_config=None,
        ema_config=None,
        first_stage_key: str = "surface",
        cond_stage_key: str = "image",
        scale_by_std: bool = False,
        z_scale_factor: float = 1.0,
        ckpt_path: Optional[str] = None,
        ignore_keys: Union[Tuple[str], List[str]] = (),
        torch_compile: bool = False,
        profile_with_events: bool = False,
    ):
        super().__init__()
        self.first_stage_key = first_stage_key
        self.cond_stage_key = cond_stage_key

        # optimizer config
        self.optimizer_cfg = optimizer_cfg
        self._train_image_encoder = bool(self.optimizer_cfg.get('train_image_encoder', False))

        # diffusion scheduler (training-time transport)
        self.scheduler_cfg = scheduler_cfg
        self.sampler = None
        if 'transport' in scheduler_cfg:
            self.transport = instantiate_from_config(scheduler_cfg.transport)
            self.sampler = instantiate_from_config(scheduler_cfg.sampler, transport=self.transport)
            self.sample_fn = self.sampler.sample_ode(**scheduler_cfg.sampler.ode_params)

        # models
        self.denoiser_cfg = denoiser_cfg
        self.model = instantiate_from_config(denoiser_cfg, device=None, dtype=None)
        self.cond_stage_model = instantiate_from_config(cond_stage_config)

        self.ckpt_path = ckpt_path
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        if lora_config is not None:
            from peft import LoraConfig, get_peft_model
            loraconfig = LoraConfig(
                r=lora_config.rank,
                lora_alpha=lora_config.rank,
                target_modules=lora_config.get('target_modules')
            )
            self.model = get_peft_model(self.model, loraconfig)

        # EMA (optional)
        self.ema_config = ema_config
        if self.ema_config is not None:
            if self.ema_config.ema_model == 'DSEma':
                self.model_ema = DSEma(self.model, decay=self.ema_config.ema_decay)
            else:
                self.model_ema = LitEma(self.model, decay=self.ema_config.ema_decay)
            # do not initialize EMA weights from ckpt (MoE topology may differ)
            if ckpt_path is not None:
                self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self._last_prof = {}
        self._last_batch_end = None
        self._profile_with_events = profile_with_events

        # VAE (non-trainable, used for decode)
        self.first_stage_model = instantiate_non_trainable_model(first_stage_config)

        # latent scaling
        self.scale_by_std = scale_by_std
        if scale_by_std:
            self.register_buffer("z_scale_factor", torch.tensor(z_scale_factor))
        else:
            self.z_scale_factor = z_scale_factor

        # inference pipeline
        self.image_processor_cfg = image_processor_cfg
        self.image_processor = None
        if self.image_processor_cfg is not None:
            self.image_processor = instantiate_from_config(self.image_processor_cfg)

        # lightweight FM Euler scheduler for inference
        from ...utils.schedulers import FlowMatchEulerDiscreteScheduler
        scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)

        self.pipeline_cfg = pipeline_cfg
        self.pipeline = instantiate_from_config(
            pipeline_cfg,
            vae=self.first_stage_model,
            model=self.model,
            scheduler=scheduler,
            conditioner=self.cond_stage_model,
            image_processor=self.image_processor
        )

        # compile (optional)
        self.torch_compile = torch_compile
        if self.torch_compile:
            torch.nn.Module.compile(self.model)
            # self.first_stage_model = torch.nn.Module.compile(self.first_stage_model, mode='reduce-overhead')
            # self.cond_stage_model = torch.nn.Module.compile(self.cond_stage_model)
            print('*' * 100)
            print('Compile model for acceleration')
            print('*' * 100)

    @property
    def _autocast_dtype(self):
        if self.trainer.precision == 'bf16':
            return torch.bfloat16
        elif self.trainer.precision == 16 or self.trainer.precision == '16':
            return torch.float16
        return torch.float32

    def _evt(self):
        if self._profile_with_events and torch.cuda.is_available():
            return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        return None, None

    def _sync_if(self, cond=True):
        if cond and torch.cuda.is_available():
            torch.cuda.synchronize()
            

    def _log_image_conditioning(self, batch, tag="val/cond_images", max_n=4):
        cond = batch.get('conditioning', None)
        if not (isinstance(cond, list) and len(cond) and hasattr(cond[0], 'size')):  # crude PIL check
            return
        imgs = [to_tensor(im) for im in cond[:max_n]]  # [0,1] tensors
        grid = torchvision.utils.make_grid(imgs, nrow=len(imgs), padding=2)
        try:
            # TensorBoard logger
            self.logger.experiment.add_image(tag, grid, global_step=self.global_step)
        except Exception:
            try:
                # W&B logger
                import wandb
                self.logger.experiment.log({tag: [wandb.Image(im) for im in cond[:max_n]],
                                            "global_step": self.global_step})
            except Exception:
                pass

    @contextmanager
    def ema_scope(self, context=None):
        if self.ema_config is not None and self.ema_config.get('ema_inference', False):
            self.model_ema.store(self.model)
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.ema_config is not None and self.ema_config.get('ema_inference', False):
                self.model_ema.restore(self.model)
                if context is not None:
                    print(f"{context}: Restored training weights")
                    
    def _resolve_ckpt_path(self, path: Union[str, Path]) -> Path:
        p = Path(path)
        if p.is_file():
            return p
        if p.is_dir():
            # prefer HF index, then a merged fp32, then DS rank file
            for pat in ["pytorch_model.bin.index.json",
                        "pytorch_model_fp32.bin",
                        "checkpoint/mp_rank_00_model_states.pt"]:
                c = p / pat
                if c.exists():
                    return c
            # last resort: any *.pt or *.bin under the dir
            cands = list(p.glob("**/*.pt")) + list(p.glob("**/*.bin"))
            if cands:
                return sorted(cands)[0]
        raise FileNotFoundError(f"Could not resolve a checkpoint file from: {path}")

    def _load_maybe_sharded(self, ckpt_path: Path):
        # HF sharded case: .../pytorch_model.bin.index.json
        if ckpt_path.suffixes[-2:] == ['.index', '.json'] and ckpt_path.name.endswith("pytorch_model.bin.index.json"):
            import json as _json
            with open(ckpt_path, "r") as f:
                index = _json.load(f)
            folder = ckpt_path.parent
            state = {}
            for part in sorted(set(index.get("weight_map", {}).values())):
                state.update(torch.load(folder / part, map_location="cpu", weights_only=False))
            return state  # raw state-dict
        # single file (PL .ckpt, DS merged .bin/.pt, etc.)
        return torch.load(ckpt_path, map_location="cpu", weights_only=False)


    def init_from_ckpt(self, path, ignore_keys=()):
        p = self._resolve_ckpt_path(path)       # <— accepts file, dir, or index.json
        obj = self._load_maybe_sharded(p)

        # pick a state dict
        if isinstance(obj, dict) and "state_dict" in obj:
            # full Lightning checkpoint (train-time save)
            sd = obj["state_dict"]
            # normalize keys a bit
            new_sd = {}
            for k, v in sd.items():
                k = k.replace("_forward_module.", "")
                new_sd[k] = v

            # drop ignored prefixes
            if ignore_keys:
                keep = {}
                for k, v in new_sd.items():
                    if not any(ik in k for ik in ignore_keys):
                        keep[k] = v
                new_sd = keep

            missing, unexpected = self.load_state_dict(new_sd, strict=False)
            print(f"Restored from {p} with {len(missing)} missing and {len(unexpected)} unexpected keys")
            if missing:   print(f"Missing Keys: {missing}")
            if unexpected:print(f"Unexpected Keys: {unexpected}")
            return

        if isinstance(obj, dict) and "module" in obj and isinstance(obj["module"], dict):
            sd = obj["module"]
        elif isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
            sd = obj["model"]
        else:
            sd = obj  # already a raw param_name -> tensor mapping

        # strip common prefixes and carve out ONLY the denoiser ('model.')
        denoiser_sd = {}
        for k, v in sd.items():
            k2 = k.replace("_forward_module.", "")
            if k2.startswith("module."): k2 = k2[7:]
            if k2.startswith("model."):  # typical for your Diffuser
                denoiser_sd[k2[len("model."):]] = v

        if not denoiser_sd and all(not k.startswith("model.") for k in sd.keys()):
            # if keys look already like pure denoiser weights, just pass them through
            denoiser_sd = sd

        # load strictly into the denoiser; CLIP/VAE are handled elsewhere
        missing, unexpected = self.model.load_state_dict(denoiser_sd, strict=False)
        print(f"[denoiser] loaded from {p} with {len(missing)} missing and {len(unexpected)} unexpected")
            
    def load_weights_only(self, path: str, ignore_keys: Tuple[str, ...] = ()):
        if not path:
            return
        self.init_from_ckpt(path, ignore_keys=ignore_keys)
            

    def on_before_backward(self, loss):
        # NVTX + event timing for backward
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_push("backward")
        self._bwd_s, self._bwd_e = self._evt()
        if self._bwd_s is not None:
            self._bwd_s.record()
        return super().on_before_backward(loss)

    def on_after_backward(self):
        if hasattr(self, "_bwd_s") and self._bwd_s is not None:
            self._bwd_e.record(); self._bwd_e.synchronize()
            self._last_prof["bwd_time"] = self._bwd_s.elapsed_time(self._bwd_e) / 1000.0
            self._bwd_s = self._bwd_e = None
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_pop()

        from pytorch_lightning.strategies import DeepSpeedStrategy
        if isinstance(self.trainer.strategy, DeepSpeedStrategy):
            return  # grads are sharded/managed, p.grad None is expected

        if self.trainer.global_rank == 0 and (self.global_step % 20 == 0):
            unused = [n for n,p in self.model.named_parameters()
                      if p.requires_grad and p.grad is None]
            if unused:
                print(f"[WARNING] {len(unused)} parameters were not used in loss computation.")
                print(f"[DEBUG] Unused params this step: {unused[:25]}{' ...' if len(unused)>25 else ''}")

    def on_load_checkpoint(self, checkpoint):
        # keep EMA tensors in checkpoint shape to satisfy PL strictness
        for key in self.state_dict().keys():
            if key.startswith("model_ema") and key not in checkpoint["state_dict"]:
                checkpoint["state_dict"][key] = self.state_dict()[key]

    def on_fit_start(self) -> None:
        # freeze/unfreeze conditioner
        for p in self.cond_stage_model.parameters():
            p.requires_grad = self._train_image_encoder

        total = sum(p.numel() for p in self.cond_stage_model.parameters())
        trainable = sum(p.numel() for p in self.cond_stage_model.parameters() if p.requires_grad)
        print(f"[on_fit_start] Conditioner trainable params: {trainable}/{total} "
            f"(train_image_encoder={self._train_image_encoder})")

        # align LR schedule when doing weights-only resume (init_ckpt)
        resume_step = getattr(self, "resume_step", None)
        real_resume = getattr(self.trainer, "ckpt_path", None) is not None  # True if --ckpt_path is used
        if (resume_step is not None) and (not real_resume):
            step = int(resume_step)
            lrs_cfgs = getattr(self.trainer, "lr_scheduler_configs", None) or []
            for cfg in lrs_cfgs:
                sch = cfg.scheduler
                if hasattr(sch, "last_epoch"):
                    sch.last_epoch = max(step - 1, 0)
            print(f"[on_fit_start] Set LR schedulers last_epoch -> {max(step-1, 0)}")

        


    def configure_optimizers(self) -> Tuple[List, List]:
        lr = self.learning_rate

        params_list = []
        trainable_parameters = list(self.model.parameters())
        params_list.append({'params': trainable_parameters, 'lr': lr})

        no_decay = ['bias', 'norm.weight', 'norm.bias', 'norm1.weight', 'norm1.bias', 'norm2.weight', 'norm2.bias']

        if self.optimizer_cfg.get('train_image_encoder', False):
            image_encoder_parameters = list(self.cond_stage_model.named_parameters())
            image_encoder_parameters_decay = [param for name, param in image_encoder_parameters
                                              if not any((nd in name) for nd in no_decay)]
            image_encoder_parameters_nodecay = [param for name, param in image_encoder_parameters
                                                if any((nd in name) for nd in no_decay)]
            # filter trainable
            image_encoder_parameters_decay = [p for p in image_encoder_parameters_decay if p.requires_grad]
            image_encoder_parameters_nodecay = [p for p in image_encoder_parameters_nodecay if p.requires_grad]

            print(f"Image Encoder Params: {len(image_encoder_parameters_decay)} decay, ")
            print(f"Image Encoder Params: {len(image_encoder_parameters_nodecay)} nodecay, ")

            image_encoder_lr = self.optimizer_cfg.get('image_encoder_lr', None)
            image_encoder_lr_multiply = self.optimizer_cfg.get('image_encoder_lr_multiply', 1.0)
            image_encoder_lr = image_encoder_lr if image_encoder_lr is not None else lr * image_encoder_lr_multiply
            params_list.append({'params': image_encoder_parameters_decay, 'lr': image_encoder_lr, 'weight_decay': 0.05})
            params_list.append({'params': image_encoder_parameters_nodecay, 'lr': image_encoder_lr, 'weight_decay': 0.0})

        optimizer = instantiate_from_config(self.optimizer_cfg.optimizer, params=params_list, lr=lr)

        # support both dict-like and object-like cfgs
        has_scheduler = (hasattr(self.optimizer_cfg, 'scheduler') and self.optimizer_cfg.scheduler is not None) or \
                        (isinstance(self.optimizer_cfg, dict) and 'scheduler' in self.optimizer_cfg)

        if has_scheduler:
            scheduler_cfg = self.optimizer_cfg.scheduler if hasattr(self.optimizer_cfg, 'scheduler') else self.optimizer_cfg['scheduler']
            scheduler_func = instantiate_from_config(
                scheduler_cfg,
                max_decay_steps=self.trainer.max_steps,
                lr_max=lr
            )
            scheduler = {
                "scheduler": lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler_func.schedule),
                "interval": "step",
                "frequency": 1
            }
            schedulers = [scheduler]
        else:
            schedulers = []
        return [optimizer], schedulers

    def optimizer_step(self, *args, **kwargs):
        use_events = self._profile_with_events and torch.cuda.is_available()

        # Do not pre-sync; measure just the optimizer section.
        if use_events:
            with nvtx("optimizer_step"):
                s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
                s.record()
                out = super().optimizer_step(*args, **kwargs)
                e.record(); e.synchronize()
                self._last_prof["opt_time"] = s.elapsed_time(e) / 1000.0  # seconds
                return out
        else:
            t0 = time.perf_counter()
            out = super().optimizer_step(*args, **kwargs)
            self._last_prof["opt_time"] = time.perf_counter() - t0
            return out

    def on_validation_epoch_end(self):
        # ensure data_time for the next batch excludes validation time
        self._last_batch_end = time.perf_counter()

    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx):
        # Reset per-step prof and compute data_time using the previous end anchor.
        now = time.perf_counter()
        # data_time should be "time since last train batch end"; if None (first batch), set 0.
        data_time = 0.0 if self._last_batch_end is None else float(now - self._last_batch_end)
        self._last_prof = {"data_time": data_time}
        self._step_start = now

        if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 \
           and batch_idx == 0 and self.ckpt_path is None:
            print("### USING STD-RESCALING ###")
            if 'latents' in batch:
                z_q = batch['latents']
            elif 'surface' in batch:
                surf = batch['surface'].to(self.device, non_blocking=True)
                z_q = self.first_stage_model.encode(surf, sample_posterior=True)
            else:
                raise ValueError("No latents or surface found in batch")
            z = z_q.detach()
            std = z.flatten().std()
            if self.trainer.num_devices > 1:
                std = self.trainer.strategy.reduce(std, reduce_op='mean')
            std = torch.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)
            std = torch.clamp(std, min=1e-6)
            if hasattr(self, "z_scale_factor"):
                del self.z_scale_factor
            self.register_buffer("z_scale_factor", (1. / std))
            print(f"setting self.z_scale_factor to {self.z_scale_factor}")
            print("### USING STD-RESCALING ###")
            
        if hasattr(self.model, "_btimer") and self.model._btimer is not None:
            self.model._btimer.use_nvtx = (self.global_step % 20) == 0

    def on_train_batch_end(self, *args, **kwargs):
        if self._profile_with_events and torch.cuda.is_available():
            torch.cuda.synchronize()
            if hasattr(self, "_last_events"):
                for name, (s, e) in self._last_events.items():
                    if s and e:
                        self._last_prof[f"{name}_time"] = s.elapsed_time(e) / 1000.0
                del self._last_events # clean up
                
        step_end = time.perf_counter()
        if hasattr(self, "_step_start") and self._step_start is not None:
            self._last_prof["step_time"] = float(step_end - self._step_start)
        else:
            self._last_prof["step_time"] = 0.0
        # anchor for next data_time
        self._last_batch_end = step_end

        if self.ema_config is not None:
            self.model_ema(self.model)

        if self._profile_with_events:
            # This logging logic should now work correctly with the accurate times
            known_time = (
                self._last_prof.get("data_time", 0.0)
                + self._last_prof.get("cond_time", 0.0)
                + self._last_prof.get("encode_time", 0.0)
                + self._last_prof.get("fwd_time", 0.0)
                + self._last_prof.get("bwd_time", 0.0)
                + self._last_prof.get("opt_time", 0.0)
            )
            other_time = self._last_prof.get("step_time", 0.0) - known_time

            metrics = {
                "train/step_time": self._last_prof.get("step_time", 0.0),
                "train/data_time": self._last_prof.get("data_time", 0.0),
                "train/cond_time": self._last_prof.get("cond_time", 0.0),
                "train/encode_time": self._last_prof.get("encode_time", 0.0),
                "train/fwd_time": self._last_prof.get("fwd_time", 0.0),
                "train/bwd_time": self._last_prof.get("bwd_time", 0.0),
                "train/opt_time": self._last_prof.get("opt_time", 0.0),
                "train/other_time": max(0.0, other_time),
            }
            self.log("train/step_time", metrics["train/step_time"], prog_bar=True, logger=True, on_step=True, rank_zero_only=True)
            self.log_dict({k: v for k, v in metrics.items() if k != "train/step_time"},
                            prog_bar=False, logger=True, on_step=True, on_epoch=False, rank_zero_only=True)
        else:
            self.log("train/step_time", self._last_prof.get("step_time", 0.0),
                        prog_bar=True, logger=True, on_step=True, rank_zero_only=True)
            
        if hasattr(self, "model") and hasattr(self.model, "collect_block_timings"):
            if (self.global_step % 20) == 0:
                bt = self.model.collect_block_timings(reset=True)
                if bt:
                    self.log_dict(
                        {f"timing/{k}": v for k, v in bt.items()},
                        prog_bar=True, logger=True, on_step=True, rank_zero_only=True
                    )

    def on_train_epoch_start(self) -> None:
        # reset data_time anchor at epoch boundary
        self._last_batch_end = time.perf_counter()

    def forward(self, batch):
        cond_grad_ctx = nullcontext() if self._train_image_encoder else torch.no_grad()

        s_cond, e_cond = self._evt()
        s_encode, e_encode = self._evt()
        s_fwd, e_fwd = self._evt()
        
        # conditioning
        with torch.autocast(device_type="cuda", dtype=self._autocast_dtype):
            with cond_grad_ctx:
                with nvtx("cond"):
                    if s_cond is not None: s_cond.record()
                    contexts = self.cond_stage_model(conditioning_data=batch.get('conditioning'))
                    # Ensure contexts match model dtype/device for fast kernels
                    def _cast_ctx(x, dtype, device):
                        if torch.is_tensor(x):
                            return x.to(device=device, dtype=dtype)
                        if isinstance(x, dict):
                            return {k: _cast_ctx(v, dtype, device) for k, v in x.items()}
                        return x
                    contexts = _cast_ctx(contexts, dtype=self.model.dtype, device=self.device)
                    if e_cond is not None: e_cond.record()

        # fetch and (optionally) scale latents (make device transfer explicit)
        with torch.autocast(device_type="cuda", dtype=self._autocast_dtype):
            with torch.no_grad():
                with nvtx("encode"):
                    if s_encode is not None: s_encode.record()
                    if 'latents' in batch:
                        latents = batch['latents'].to(self.device, non_blocking=True)
                    elif 'surface' in batch:
                        surf = batch['surface'].to(self.device, non_blocking=True)
                        latents = self.first_stage_model.encode(surf, sample_posterior=True)
                    else:
                        raise ValueError("No latents or surface found in batch")
                    if hasattr(self, "z_scale_factor"):
                        latents = self.z_scale_factor * latents
                    if e_encode is not None: e_encode.record()

        # flow-matching loss
        with torch.autocast(device_type="cuda", dtype=self._autocast_dtype):
            with nvtx("fm_fwd"):
                if s_fwd is not None: s_fwd.record()
                loss = self.transport.training_losses(self.model, latents, dict(contexts=contexts))["loss"].mean()
                if e_fwd is not None: e_fwd.record()
                    
        if self._profile_with_events:
            self._last_events = {
                "cond": (s_cond, e_cond),
                "encode": (s_encode, e_encode),
                "fwd": (s_fwd, e_fwd)
            }

        return loss

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        loss = self.forward(batch)
        split = 'train'
        self.log_dict(
            {
                f"{split}/simple": loss.detach(),
                f"{split}/total_loss": loss.detach(),
                f"{split}/lr_abs": self.optimizers().param_groups[0]['lr'],
            },
            prog_bar=False, logger=True, on_step=True, on_epoch=False, sync_dist=False, rank_zero_only=True
        )
        if (self.global_step % 5000) == 0:
            self._log_image_conditioning(batch, tag="val/cond_images", max_n=4)
        return loss

    @torch.inference_mode()
    @dynamo.disable()
    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        loss = self.forward(batch)
        split = 'val'
        self.log_dict(
            {
                f"{split}/simple": loss.detach(),
                f"{split}/total_loss": loss.detach(),
                f"{split}/lr_abs": self.optimizers().param_groups[0]['lr'],
            },
            prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=False, rank_zero_only=True
        )
        if (self.global_step % 5000) == 0:
            self._log_image_conditioning(batch, tag="val/cond_images", max_n=4)
        if "view_idx" in batch:
            self.log("val/view_idx", batch["view_idx"].float().mean(), on_step=True, prog_bar=False)
        return loss

    @torch.no_grad()
    @dynamo.disable()
    def sample(
        self,
        conditioning=None,
        batch_size: int = 1,
        generator: torch.Generator = None,
        output_type: str = 'trimesh',
        **kwargs
    ):
        """
        Generate 3D assets via the configured pipeline.

        Args:
            conditioning: None | str | List[str] | Tensor[int]
            batch_size:  number of samples
            generator:   optional RNG
            output_type: 'trimesh' (preferred). For BC: accepts 'latents2mesh' and maps it to 'trimesh'.
            **kwargs:    forwarded to pipeline (bounds, mc_level, num_chunks, octree_resolution, ...)
        """
        was_training = self.cond_stage_model.training
        self.cond_stage_model.eval()

        if generator is None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(0)

        try:
            with self.ema_scope("Sample"):
                with torch.amp.autocast(device_type='cuda', dtype=self._autocast_dtype):
                    self.pipeline.device = self.device
                    self.pipeline.dtype = self.dtype
                    print("### USING PIPELINE FOR SAMPLING ###")

                    outputs = self.pipeline(
                        conditioning=conditioning,
                        batch_size=batch_size,
                        generator=generator,
                        output_type=output_type,
                        z_scale_factor=getattr(self, "z_scale_factor", 1.0),
                        **kwargs
                    )
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"An unexpected error occurred during sampling: {e}")
            outputs = [None] * batch_size
        finally:
            self.cond_stage_model.train(was_training)

        # BC with callbacks expecting a list-of-lists
        return [outputs]
