from __future__ import annotations
from typing import List
import torch
from hy3dshape.utils.metrics.encoders.pointnet_impl.pointnet import PointNetEncoderLocal as _PN
from . import IFeatureEncoder, _file_fingerprint

class PointNetAdapter(IFeatureEncoder):
    def __init__(self, ckpt_path: str, width_mult: int, device_batch_size: int,
                 normal_channel: bool, devices: List[torch.device]):
        self._inner = _PN(
            ckpt_path=ckpt_path,
            width_mult=width_mult,
            device_batch_size=device_batch_size,
            normal_channel=normal_channel,
            devices=devices,
        )
        self.name = "pointnet"
        self.version = f"w{int(width_mult)}"
        self.feature_dim = getattr(self._inner, "feature_dim", 1024)
        self.ckpt_fingerprint = _file_fingerprint(ckpt_path)

    def to(self, device: torch.device) -> "PointNetAdapter":
        # Underlying encoder already binds to devices; no-op.
        return self

    def encode_np(self, pts_bxn3_np):
        return self._inner.encode_np(pts_bxn3_np)
