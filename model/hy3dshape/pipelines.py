import os
import importlib
import inspect
from typing import List, Optional, Union, Dict

import torch
import trimesh
import yaml
import numpy as np
from tqdm import tqdm
from PIL import Image

from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils.import_utils import is_accelerate_available, is_accelerate_version

from .utils import logger, synchronize_timer, smart_load_model

def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Standard helper to set and fetch timesteps (or sigmas) from a diffusers scheduler.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")

    accepts = set(inspect.signature(scheduler.set_timesteps).parameters.keys())

    if timesteps is not None:
        if "timesteps" not in accepts:
            raise ValueError(f"{scheduler.__class__} does not support custom timesteps.")
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        return scheduler.timesteps, len(scheduler.timesteps)

    if sigmas is not None:
        if "sigmas" not in accepts:
            raise ValueError(f"{scheduler.__class__} does not support custom sigmas.")
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        return scheduler.timesteps, len(scheduler.timesteps)

    scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
    return scheduler.timesteps, num_inference_steps


@synchronize_timer('Export to trimesh')
def export_to_trimesh(mesh_output):
    """
    Convert Latent2MeshOutput -> trimesh.Trimesh (or list of them).
    """
    def _one(mo):
        if mo is None:
            return None
        mo.mesh_f = mo.mesh_f[:, ::-1]
        return trimesh.Trimesh(mo.mesh_v, mo.mesh_f)

    if isinstance(mesh_output, list):
        return [_one(m) for m in mesh_output]
    return _one(mesh_output)


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config: Dict, **kwargs):
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")
    cls = get_obj_from_str(config["target"])
    params = config.get("params", dict())
    kwargs.update(params)
    instance = cls(**kwargs)
    return instance

class Hunyuan3DDiTPipeline:
    model_cpu_offload_seq = "conditioner->model->vae"
    _exclude_from_cpu_offload = []

    def __init__(
        self,
        vae,
        model,
        scheduler,
        conditioner,
        image_processor,
        device='cuda',
        dtype=torch.float16,
        **kwargs
    ):
        self.vae = vae
        self.model = model
        self.scheduler = scheduler
        self.conditioner = conditioner
        self.image_processor = image_processor
        self.kwargs = kwargs
        self.to(device, dtype)

    @classmethod
    @synchronize_timer('Hunyuan3DDiTPipeline Model Loading')
    def from_single_file(
        cls,
        ckpt_path,
        config_path,
        device='cuda',
        dtype=torch.float16,
        use_safetensors=None,
        **kwargs,
    ):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        if use_safetensors:
            ckpt_path = ckpt_path.replace('.ckpt', '.safetensors')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Model file {ckpt_path} not found")
        logger.info(f"Loading model from {ckpt_path}")

        if use_safetensors:
            import safetensors.torch
            safetensors_ckpt = safetensors.torch.load_file(ckpt_path, device='cpu')
            ckpt = {}
            for key, value in safetensors_ckpt.items():
                model_name = key.split('.')[0]
                new_key = key[len(model_name) + 1:]
                if model_name not in ckpt:
                    ckpt[model_name] = {}
                ckpt[model_name][new_key] = value
        else:
            ckpt = torch.load(ckpt_path, map_location='cpu')

        model = instantiate_from_config(config['model'])
        model.load_state_dict(ckpt['model'])
        
        vae = instantiate_from_config(config['vae'])
        vae.load_state_dict(ckpt['vae'], strict=False)
        
        conditioner = instantiate_from_config(config['conditioner'])
        if 'conditioner' in ckpt:
            conditioner.load_state_dict(ckpt['conditioner'])
            
        image_processor = instantiate_from_config(config['image_processor'])
        scheduler = instantiate_from_config(config['scheduler'])

        return cls(
            vae=vae, model=model, scheduler=scheduler, 
            conditioner=conditioner, image_processor=image_processor,
            device=device, dtype=dtype, **kwargs
        )

    @classmethod
    def from_pretrained(
        cls,
        model_path,
        device='cuda',
        dtype=torch.float16,
        use_safetensors=False,
        variant='fp16',
        subfolder='hunyuan3d-dit-v2-1',
        **kwargs,
    ):
        kwargs['from_pretrained_kwargs'] = dict(
            model_path=model_path,
            subfolder=subfolder,
            use_safetensors=use_safetensors,
            variant=variant,
            dtype=dtype,
            device=device,
        )
        config_path, ckpt_path = smart_load_model(
            model_path,
            subfolder=subfolder,
            use_safetensors=use_safetensors,
            variant=variant
        )
        return cls.from_single_file(
            ckpt_path,
            config_path,
            device=device,
            dtype=dtype,
            use_safetensors=use_safetensors,
            **kwargs
        )

    def compile(self):
        self.vae = torch.compile(self.vae)
        self.model = torch.compile(self.model)
        self.conditioner = torch.compile(self.conditioner)

    def enable_flashvdm(self, *args, **kwargs):
        logger.info("enable_flashvdm: no-op in this build.")

    def set_surface_extractor(self, mc_algo):
        return

    def to(self, device=None, dtype=None):
        if dtype is not None:
            self.dtype = dtype
            self.vae.to(dtype=dtype)
            self.model.to(dtype=dtype)
            try:
                self.conditioner.to(dtype=dtype)
            except Exception:
                pass
        if device is not None:
            self.device = torch.device(device)
            self.vae.to(device)
            self.model.to(device)
            self.conditioner.to(device)

    @property
    def _execution_device(self):
        for name, model in self.components.items():
            if not isinstance(model, torch.nn.Module) or name in self._exclude_from_cpu_offload:
                continue
            if not hasattr(model, "_hf_hook"):
                return self.device
            for module in model.modules():
                if (hasattr(module, "_hf_hook") and 
                    hasattr(module._hf_hook, "execution_device") and 
                    module._hf_hook.execution_device is not None):
                    return torch.device(module._hf_hook.execution_device)
        return self.device

    def enable_model_cpu_offload(self, gpu_id: Optional[int] = None, device: Union[torch.device, str] = "cuda"):
        if self.model_cpu_offload_seq is None:
            raise ValueError("Model CPU offload cannot be enabled because no `model_cpu_offload_seq` is set.")

        if is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"):
            from accelerate import cpu_offload_with_hook
        else:
            raise ImportError("`enable_model_cpu_offload` requires `accelerate v0.17.0` or higher.")

        torch_device = torch.device(device)
        device_index = torch_device.index

        if gpu_id is not None and device_index is not None:
            raise ValueError(f"You passed both `gpu_id`={gpu_id} and device with index `device`={device}. Choose one.")

        self._offload_gpu_id = gpu_id or device_index or getattr(self, "_offload_gpu_id", 0)
        device_type = torch_device.type
        device = torch.device(f"{device_type}:{self._offload_gpu_id}")

        if self.device.type != "cpu":
            self.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        all_model_components = {k: v for k, v in self.__dict__.items() if isinstance(v, torch.nn.Module)}

        self._all_hooks = []
        hook = None
        for model_str in self.model_cpu_offload_seq.split("->"):
            model = all_model_components.pop(model_str, None)
            if not isinstance(model, torch.nn.Module):
                continue
            _, hook = cpu_offload_with_hook(model, device, prev_module_hook=hook)
            self._all_hooks.append(hook)

        for name, model in all_model_components.items():
            if not isinstance(model, torch.nn.Module):
                continue
            if name in self._exclude_from_cpu_offload:
                model.to(device)
            else:
                _, hook = cpu_offload_with_hook(model, device)
                self._all_hooks.append(hook)

    def maybe_free_model_hooks(self):
        if not hasattr(self, "_all_hooks") or len(self._all_hooks) == 0:
            return
        for hook in self._all_hooks:
            hook.offload()
            hook.remove()
        self.enable_model_cpu_offload()

    def prepare_latents(self, batch_size, dtype, device, generator, latents=None):
        if not hasattr(self.model, 'max_seq_len') or not hasattr(self.model, 'in_channels'):
            raise AttributeError("The denoiser model must have 'max_seq_len' and 'in_channels' attributes.")
        
        shape = (batch_size, self.model.max_seq_len, self.model.in_channels)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(f"You passed {len(generator)} generators, but batch size is {batch_size}.")

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        return latents * getattr(self.scheduler, 'init_noise_sigma', 1.0)

    def prepare_image(self, image, mask=None) -> dict:
        if isinstance(image, torch.Tensor) and isinstance(mask, torch.Tensor):
            return {'image': image, 'mask': mask}

        if isinstance(image, str) and not os.path.exists(image):
            raise FileNotFoundError(f"Couldn't find image at path {image}")

        if not isinstance(image, list):
            image = [image]

        outputs = []
        for img in image:
            output = self.image_processor(img)
            outputs.append(output)

        cond_input = {k: [] for k in outputs[0].keys()}
        for output in outputs:
            for key, value in output.items():
                cond_input[key].append(value)
        for key, value in cond_input.items():
            if isinstance(value[0], torch.Tensor):
                cond_input[key] = torch.cat(value, dim=0)

        return cond_input

    def _export(self, latents, output_type='trimesh', box_v=1.01, mc_level=0.0, num_chunks=20000, octree_resolution=256, mc_algo='mc', enable_pbar=True):
        latents = (1. / getattr(self.vae, "scale_factor", 1.0)) * latents if hasattr(self.vae, "scale_factor") else latents
        try:
            latents = self.vae(latents)
        except TypeError:
            pass

        outputs = self.vae.latents2mesh(
            latents,
            bounds=box_v,
            mc_level=mc_level,
            num_chunks=num_chunks,
            octree_resolution=octree_resolution,
            enable_pbar=enable_pbar,
        )

        if output_type == 'trimesh':
            outputs = export_to_trimesh(outputs)
        return outputs



class Hunyuan3DDiTFlowMatchingPipeline(Hunyuan3DDiTPipeline):

    @torch.inference_mode()
    def __call__(
        self,
        image: Union[str, List[str], Image.Image, dict, List[dict], torch.Tensor] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        eta: float = 0.0,
        guidance_scale: float = 5.0,
        generator=None,
        box_v=1.01,
        octree_resolution=384,
        mc_level=0.0,
        mc_algo=None,
        num_chunks=8000,
        output_type: Optional[str] = "trimesh",
        enable_pbar=True,
        mask=None,
        **kwargs,
    ):
        self.set_surface_extractor(mc_algo)
        device = self.device
        dtype = self.dtype
        do_classifier_free_guidance = guidance_scale >= 0 and not (
            hasattr(self.model, 'guidance_embed') and self.model.guidance_embed is True
        )

        cond_inputs = self.prepare_image(image, mask)
        image = cond_inputs.pop('image')
        cond = self.conditioner(image=image, **cond_inputs)
        batch_size = image.shape[0]

        sigmas = np.linspace(0, 1, num_inference_steps) if sigmas is None else sigmas
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
        )
        latents = self.prepare_latents(batch_size, dtype, device, generator)

        guidance = None
        if hasattr(self.model, 'guidance_embed') and self.model.guidance_embed is True:
            guidance = torch.tensor([guidance_scale] * batch_size, device=device, dtype=dtype)

        with synchronize_timer('Diffusion Sampling'):
            for t in tqdm(timesteps, disable=not enable_pbar, desc="Diffusion Sampling:"):
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                timestep = t.expand(latent_model_input.shape[0]).to(latents.dtype)
                if hasattr(self.scheduler, "config") and hasattr(self.scheduler.config, "num_train_timesteps"):
                    timestep = timestep / self.scheduler.config.num_train_timesteps

                noise_pred = self.model(latent_model_input, timestep, cond, guidance=guidance)

                if do_classifier_free_guidance:
                    noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                outputs = self.scheduler.step(noise_pred, t, latents)
                latents = outputs.prev_sample

        return self._export(
            latents,
            output_type,
            box_v, mc_level, num_chunks, octree_resolution, mc_algo,
            enable_pbar=enable_pbar,
        )


# Custom pipeline
class CustomFlowMatchingPipeline(Hunyuan3DDiTPipeline):
    def __init__(self, vae, model, scheduler, conditioner, image_processor, 
                 pipeline_output: str = 'mesh', **kwargs):
        super().__init__(vae, model, scheduler, conditioner, image_processor, **kwargs)
        self.pipeline_output = pipeline_output

    @torch.inference_mode()
    def __call__(self, conditioning=None, batch_size: int = 1, num_inference_steps: int = 50,
                guidance_scale: float = 5.0, generator=None, output_type: str = "trimesh", z_scale_factor=1.0, **kwargs):
        device, dtype = self.device, self.dtype
        do_cfg = guidance_scale > 1.0 and conditioning is not None

        def _ddp_info():
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                try:
                    return torch.distributed.get_rank(), torch.distributed.get_world_size()
                except Exception:
                    pass
            return 0, 1
        rank, world = _ddp_info()
        show_progress = bool(kwargs.pop("show_progress", True))

        # Prepare latents
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        latents = self.prepare_latents(batch_size, dtype, device, generator)

        # Prepare conditioning
        if conditioning is None:
            cond = self.conditioner(conditioning_data=None)
        else:
            cond = self.conditioner(conditioning_data=conditioning)

        if do_cfg:
            uncond = self.conditioner.unconditional_embedding(batch_size, device=device)
            def cat_ctx(a, b):
                if isinstance(a, torch.Tensor):
                    return torch.cat([a, b], dim=0)
                return {k: cat_ctx(a[k], b[k]) for k in a}
            ctx = cat_ctx(cond, uncond)
        else:
            ctx = cond

        # FM sampling loop
        _disable_bar = (not show_progress) or (world > 1 and rank != 0)
        _indices = range(timesteps.shape[0]) if torch.is_tensor(timesteps) else range(len(timesteps))
        
        for i in tqdm(_indices, desc=f"Diffusion Sampling [rank{rank}]", leave=False, disable=_disable_bar):
            t = timesteps[i] if torch.is_tensor(timesteps) else timesteps[i]
            if do_cfg:
                latent_model_input = torch.cat([latents, latents], dim=0)
            else:
                latent_model_input = latents

            t_model = (t / self.scheduler.config.num_train_timesteps).expand(latent_model_input.shape[0]).to(dtype)
            noise_pred = self.model(latent_model_input, t_model, contexts=ctx)

            if do_cfg:
                pred_cond, pred_uncond = noise_pred.chunk(2)
                noise_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        if z_scale_factor != 1.0:
            latents = latents / z_scale_factor

        if self.pipeline_output == 'mesh':
            return self._export_latents_to_mesh(latents, output_type=output_type, **kwargs)
        
        elif self.pipeline_output == 'image':
            if output_type == "trimesh":
                output_type = "pil"
            return self._export_latents_to_image(latents, output_type=output_type, **kwargs)
            
        else:
            raise ValueError(f"Unsupported pipeline output_type: {self.pipeline_output}")

    def _export_latents_to_mesh(self, latents, output_type='trimesh', bounds=1.01, mc_level=0.0, num_chunks=50000, octree_resolution=256, mc_algo=None, enable_pbar=True, **kwargs):
        if 'box_v' in kwargs and 'bounds' not in kwargs:
            kwargs['bounds'] = kwargs.pop('box_v')
        if 'octree_depth' in kwargs and 'octree_resolution' not in kwargs:
            depth = int(kwargs.pop('octree_depth'))
            kwargs['octree_resolution'] = 1 << depth
        kwargs.setdefault('bounds', bounds)
        kwargs.setdefault('mc_level', mc_level)
        kwargs.setdefault('octree_resolution', octree_resolution)
        kwargs.setdefault('num_chunks', num_chunks)
        kwargs.setdefault('enable_pbar', enable_pbar)
        
        if mc_algo is not None and hasattr(self, 'set_surface_extractor'):
            try: self.set_surface_extractor(mc_algo)
            except Exception as e:
                logger.warning(f"set_surface_extractor({mc_algo}) failed: {e}")

        try:
            outputs = self.vae.latents2mesh(latents, **kwargs)
        except ValueError as e:
            if "Surface level must be within volume data range" in str(e):
                bsz = latents.shape[0]
                return [None] * bsz
            raise e

        if output_type == 'trimesh':
            outputs = export_to_trimesh(outputs)
        return outputs

    def _export_latents_to_image(self, latents: torch.Tensor, output_type: str = "pil", **kwargs):
        if not hasattr(self.model, 'max_seq_len') or not hasattr(self.model, 'in_channels'):
            raise AttributeError("The denoiser model must have 'max_seq_len' and 'in_channels' attributes.")
        
        num_tokens = self.model.max_seq_len
        latent_channels = self.model.in_channels
        
        grid_size = int(num_tokens**0.5)
        if grid_size * grid_size != num_tokens:
            raise ValueError(f"Cannot form a square grid from {num_tokens} tokens.")

        B, T, C = latents.shape
        if T != num_tokens or C != latent_channels:
            raise ValueError(f"Latent shape mismatch. Got {latents.shape}, expected B x {num_tokens} x {latent_channels}")
            
        latents_grid = latents.permute(0, 2, 1).reshape(B, latent_channels, grid_size, grid_size)
        images = self.vae.decode(latents_grid)
        images = (images / 2 + 0.5).clamp(0, 1)

        if output_type == "pil":
            from torchvision.transforms.functional import to_pil_image
            return [to_pil_image(img) for img in images]
        
        return images