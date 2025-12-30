from __future__ import annotations
import os
from typing import List, Optional, Union, Tuple

import numpy as np
import torch

try:
    from hy3dshape.utils.metrics.encoders.pointnet_impl.pointnet2_cls_ssg import get_model as _get_model
except Exception as e:
    raise ImportError(
        "Missing local pointnet2_cls_ssg.py in hy3dshape/utils/metrics/. "
        "Please place the vendored implementation there."
    ) from e


def _all_visible_devices() -> List[Union[str, torch.device]]:
    if torch.cuda.is_available():
        return [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    return ["cpu"]


def normalize_bbox_unit_sphere(pc: np.ndarray) -> np.ndarray:
    assert pc.ndim == 3 and pc.shape[-1] == 3, f"expected (N,P,3), got {pc.shape}"
    vmin = pc.min(axis=1, keepdims=True)
    vmax = pc.max(axis=1, keepdims=True)
    shifts = (vmax + vmin) / 2.0
    v = pc - shifts
    norms = np.linalg.norm(v, axis=2, keepdims=True)
    max_norm = norms.max(axis=1, keepdims=True)
    scale = 1.0 / np.clip(max_norm, 1e-12, None)
    v = v * scale
    return v.astype(np.float32)


class PointNetEncoderLocal:

    def __init__(
        self,
        ckpt_path: str,
        devices: Optional[List[Union[str, torch.device]]] = None,
        device_batch_size: int = 64,
        width_mult: int = 2,
        normal_channel: bool = False,
        normalize: bool = False,
    ):
        self.ckpt_path = ckpt_path
        self.devices = devices or _all_visible_devices()
        self.device_batch_size = int(device_batch_size)
        self.width_mult = int(width_mult)
        self.normal_channel = bool(normal_channel)
        self.normalize_inside = bool(normalize)

        sd = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]

        cleaned = {}
        for k, v in sd.items():
            k2 = k
            if k2.startswith("module."):
                k2 = k2[len("module.") :]
            cleaned[k2] = v
        sd = cleaned

        self.models: List[torch.nn.Module] = []
        for dev in self.devices:
            m = _get_model(num_class=40, normal_channel=self.normal_channel, width_mult=self.width_mult)
            missing, unexpected = m.load_state_dict(sd, strict=False)
            if missing or unexpected:
                print(f"[PointNet] loaded with {len(missing)} missing and {len(unexpected)} unexpected keys")
            m.to(dev)
            m.eval()
            self.models.append(m)

        # feature dimensionality = 256 * width_mult (Point-E uses width_mult=2 => 512)
        self.feature_dim = 256 * self.width_mult

    @torch.no_grad()
    def encode_np(self, pcs: np.ndarray) -> np.ndarray:
        assert pcs.ndim == 3 and pcs.shape[-1] == 3, f"expected (N,P,3), got {pcs.shape}"
        pcs = pcs.astype(np.float32)
        if self.normalize_inside:
            pcs = normalize_bbox_unit_sphere(pcs)

        out_chunks: List[np.ndarray] = []
        n = pcs.shape[0]
        num_devices = max(1, len(self.devices))

        for start in range(0, n, self.device_batch_size * num_devices):
            jobs: List[Tuple[int, torch.Tensor, torch.device]] = []
            for d_i, dev in enumerate(self.devices):
                s = start + d_i * self.device_batch_size
                if s >= n:
                    break
                e = min(n, s + self.device_batch_size)
                x = torch.from_numpy(pcs[s:e]).permute(0, 2, 1).to(dev, dtype=torch.float32)  # (B,3,P)
                jobs.append((d_i, x, dev))

            for (d_i, x, dev) in jobs:
                logits, _, feats = self.models[d_i](x, features=True)  # feats: (B, 256*width_mult)
                out_chunks.append(feats.detach().cpu().numpy())

        return np.concatenate(out_chunks, axis=0)
