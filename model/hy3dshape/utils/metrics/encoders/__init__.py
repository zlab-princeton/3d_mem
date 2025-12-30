# hy3dshape/utils/metrics/encoders/__init__.py
from __future__ import annotations
from typing import Any, Dict
import os
import torch

def _file_fingerprint(path: str) -> str:
    try:
        st = os.stat(path)
        return f"{st.st_size}-{int(st.st_mtime)}"
    except Exception:
        return "none"
    
    
    
class IFeatureEncoder:
    """
    Minimal interface all encoders should expose.
    """
    name: str                # short, lower-case (e.g., "pointnet", "uni3d")
    version: str             # model/width variant string (e.g., "w2", "base")
    feature_dim: int
    ckpt_fingerprint: str    # changes when weights change

    def to(self, device: torch.device) -> "IFeatureEncoder":
        return self

    def encode_np(self, pts_bxn3_np) -> "np.ndarray":
        raise NotImplementedError

    @property
    def cache_tag(self) -> str:
        # used for ref-cache file names
        return f"{self.name}-{self.version}-{self.ckpt_fingerprint}"

def build_encoder(cfg: Dict[str, Any], device: torch.device):
    name = (cfg.get("name") or "pointnet").lower()

    if name in ("pointnet", "pn"):
        from .pointnet_adapter import PointNetAdapter
        return PointNetAdapter(
            ckpt_path=cfg.get("ckpt", cfg.get("pointnet_ckpt", "")),
            width_mult=int(cfg.get("width_mult", 2)),
            device_batch_size=int(cfg.get("device_batch_size", 64)),
            normal_channel=bool(cfg.get("normal_channel", False)),
            devices=[device],
        )

    if name in ("uni3d", "uni-3d", "uni"):
        from .uni3d_adapter import Uni3DAdapter
        return Uni3DAdapter(
            ckpt=cfg["ckpt"],
            variant=cfg.get("variant", "base"),
            n_points=cfg.get("n_points", None),            # will default to callback n_points if you set it there
            device_batch_size=int(cfg.get("device_batch_size", 64)),
            normal_channel=bool(cfg.get("normal_channel", False)),
            devices=[device],
            amp=bool(cfg.get("amp", True)),
            pc_model=cfg.get("pc_model", None),
            pretrained_pc=cfg.get("pretrained_pc", ""),
        )

    raise ValueError(f"Unknown encoder name: {name}")