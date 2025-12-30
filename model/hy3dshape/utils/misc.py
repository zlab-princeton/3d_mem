import importlib
from omegaconf import OmegaConf, DictConfig, ListConfig

import os, torch
import torch.distributed as dist
from typing import Union


def get_config_from_file(config_file: str) -> Union[DictConfig, ListConfig]:
    config_file = OmegaConf.load(config_file)

    if 'base_config' in config_file.keys():
        if config_file['base_config'] == "default_base":
            base_config = OmegaConf.create()
            # base_config = get_default_config()
        elif config_file['base_config'].endswith(".yaml"):
            base_config = get_config_from_file(config_file['base_config'])
        else:
            raise ValueError(f"{config_file} must be `.yaml` file or it contains `base_config` key.")

        config_file = {key: value for key, value in config_file if key != "base_config"}

        return OmegaConf.merge(base_config, config_file)

    return config_file


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def get_obj_from_config(config):
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")

    return get_obj_from_str(config["target"])


def instantiate_from_config(config, **kwargs):
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")

    cls = get_obj_from_str(config["target"])

    if config.get("from_pretrained", None):
        return cls.from_pretrained(
                    config["from_pretrained"], 
                    use_safetensors=config.get('use_safetensors', False),
                    variant=config.get('variant', 'fp16'))

    params = config.get("params", dict())
    # params.update(kwargs)
    # instance = cls(**params)
    kwargs.update(params)
    kwargs.pop("pretrained", None)  # don't pass this to the constructor
    instance = cls(**kwargs)

    return instance


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def _extract_vae_state_dict(sd, ckpt_key=None):
    if not isinstance(sd, dict):
        return sd
    if ckpt_key and ckpt_key in sd and isinstance(sd[ckpt_key], dict):
        return sd[ckpt_key]
    for k in ("state_dict", "model", "vae", "first_stage_model", "module", "ema_model"):
        if k in sd and isinstance(sd[k], dict):
            return sd[k]
    return sd

def _strip_prefixes(state_dict, prefixes):
    out = {}
    for k, v in state_dict.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p):]
        out[nk] = v
    return out

def _drop_if_prefixes(state_dict, drop_prefixes):
    if not drop_prefixes:
        return state_dict
    out = {}
    for k, v in state_dict.items():
        if any(k.startswith(p) for p in drop_prefixes):
            continue
        out[k] = v
    return out

def _route_into_submodule(state_dict, target):
    if not target:
        return state_dict
    routed = {}
    pref = target + "."
    for k, v in state_dict.items():
        routed[k if k.startswith(pref) else (pref + k)] = v
    return routed


def instantiate_non_trainable_model(config):
    model = instantiate_from_config(config)

    params = config.get("params", {}) or {}
    pretrained = params.get("pretrained", {}) or {}
    ckpt_path       = pretrained.get("ckpt_path", None)
    ckpt_key        = pretrained.get("ckpt_key", "model")  # your saves use {'model': ...}
    strip_prefixes  = pretrained.get("strip_prefixes", ["first_stage_model.","vae.","model.","module."])
    target_submod   = pretrained.get("target_submodule", "autoencoder")  # map into CustomShapeVAE.autoencoder
    drop_prefixes   = pretrained.get("drop_prefixes", None)  # e.g., ["latents."] if switching to query_type='point'
    if ckpt_path:
        ckpt_path = os.path.abspath(os.path.expanduser(ckpt_path))
        try:
            raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            sd  = _extract_vae_state_dict(raw, ckpt_key=ckpt_key)
            sd  = _strip_prefixes(sd, strip_prefixes)
            sd  = _drop_if_prefixes(sd, drop_prefixes)
            sd  = _route_into_submodule(sd, target_submod)

            missing, unexpected = model.load_state_dict(sd, strict=False)
            print(f"[VAE] loaded {ckpt_path} | tensors={len(sd)} | missing={len(missing)} | unexpected={len(unexpected)}")
            if missing:    print("  missing (first 12):", [m for m in list(missing)[:12]])
            if unexpected: print("  unexpected (first 12):", [u for u in list(unexpected)[:12]])
        except Exception as e:
            print(f"[VAE] Failed to load {ckpt_path}: {e}")
    model = model.eval()
    model.train = disabled_train
    for param in model.parameters():
        param.requires_grad = False

    return model


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def all_gather_batch(tensors):
    """
    Performs all_gather operation on the provided tensors.
    """
    # Queue the gathered tensors
    world_size = get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors
    tensor_list = []
    output_tensor = []
    for tensor in tensors:
        tensor_all = [torch.ones_like(tensor) for _ in range(world_size)]
        dist.all_gather(
            tensor_all,
            tensor,
            async_op=False  # performance opt
        )

        tensor_list.append(tensor_all)

    for tensor_all in tensor_list:
        output_tensor.append(torch.cat(tensor_all, dim=0))
    return output_tensor
