from __future__ import annotations
import numpy as np
import torch
from . import IFeatureEncoder, _file_fingerprint
from .uni3d_impl.uni3d_model import create_uni3d

class Uni3DAdapter(IFeatureEncoder):
    def __init__(self,
                 ckpt: str,
                 variant: str = "base",
                 n_points: int | None = None,
                 device_batch_size: int = 64,
                 normal_channel: bool = False,  # ignored; kept for config parity
                 devices: list[torch.device] | None = None,
                 amp: bool = True,
                 pc_model: str | None = None,
                 pretrained_pc: str = ""):
        self.name = "uni3d"
        self.version = variant
        self.ckpt_fingerprint = _file_fingerprint(ckpt)
        self.device_batch_size = int(device_batch_size)
        self.normal_channel = bool(normal_channel)
        self.amp = bool(amp)

        self.device = devices[0] if devices else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_points = int(n_points) if n_points is not None else 4096

        # Variant defaults; you can override via pc_model/pretrained_pc in YAML
        EVA = {
            "giant": dict(pc_model="eva_giant_patch14_560",   pc_feat_dim=1408, embed_dim=1024),
            "large": dict(pc_model="eva02_large_patch14_448", pc_feat_dim=1024, embed_dim=1024),
            "base":  dict(pc_model="eva02_base_patch14_448",  pc_feat_dim=768,  embed_dim=1024),
            "small": dict(pc_model="eva02_small_patch14_224", pc_feat_dim=384,  embed_dim=384),
            "tiny":  dict(pc_model="eva02_tiny_patch14_224",  pc_feat_dim=192,  embed_dim=192),
        }[variant]

        # Build a tiny args object for create_uni3d()
        class Cfg: ...
        cfg = Cfg()
        cfg.pc_model         = pc_model or EVA["pc_model"]
        cfg.pretrained_pc    = pretrained_pc or ""
        cfg.drop_path_rate   = 0.0
        cfg.num_points       = self.n_points
        cfg.num_group        = 512
        cfg.group_size       = 64
        cfg.pc_encoder_dim   = 512
        cfg.embed_dim        = EVA["embed_dim"]
        cfg.pc_feat_dim      = EVA["pc_feat_dim"]
        cfg.patch_dropout    = 0.0

        self.model = create_uni3d(cfg).to(self.device)
        # load checkpoint if provided
        if ckpt:
            sd = torch.load(ckpt, map_location="cpu", weights_only=False)
            # try common containers
            for key in ("state_dict", "model", "module", "ema_state_dict"):
                if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
                    sd = sd[key]
                    break
            # strip module. and ignore heads
            cleaned = {}
            for k, v in (sd.items() if isinstance(sd, dict) else []):
                if not isinstance(v, torch.Tensor): continue
                nk = k[7:] if k.startswith("module.") else k
                if any(s in nk for s in ("optimizer", "scaler", "sched", "epoch", "args")): continue
                if ".fc_norm" in nk or "visual.head" in nk: continue
                cleaned[nk] = v
            self.model.load_state_dict(cleaned, strict=False)

        self.model.eval()
        self.feature_dim = int(getattr(self.model, "embed_dim", EVA["embed_dim"]))

    @torch.inference_mode()
    def encode_np(self, pcs_bxn3: np.ndarray) -> np.ndarray:
        assert pcs_bxn3.ndim == 3 and pcs_bxn3.shape[-1] == 3, f"expected (B,N,3), got {pcs_bxn3.shape}"
        out = []
        for i in range(0, pcs_bxn3.shape[0], self.device_batch_size):
            x = pcs_bxn3[i:i+self.device_batch_size].astype(np.float32, copy=False)  # (b,N,3)
            rgb = np.zeros_like(x, dtype=np.float32)
            x6 = np.concatenate([x, rgb], axis=-1)  # (b,N,6)
            xt = torch.from_numpy(x6).to(self.device, non_blocking=True)
            if self.amp and self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    emb = self.model.encode_pc(xt)
            else:
                emb = self.model.encode_pc(xt)
            emb = torch.nn.functional.normalize(emb, dim=-1)
            out.append(emb.detach().cpu().float().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, self.feature_dim), dtype=np.float32)
