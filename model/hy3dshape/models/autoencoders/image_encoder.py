from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL
from contextlib import nullcontext
from typing import Optional, Tuple

class FluxVAE(nn.Module):
    """Thin wrapper around Diffusers AutoencoderKL with explicit scale/shift handling."""
    def __init__(
        self,
        pretrained_path: str,
        device: str | torch.device = "cuda",
        dtype: str = "bf16",              # {"bf16","fp16","fp32"}
        freeze: bool = True,
        enforce_fp32: bool = False,       # set True if you want encode/decode to run in fp32
    ):
        super().__init__()
        self.vae: AutoencoderKL = AutoencoderKL.from_pretrained(pretrained_path)
        self.register_buffer("_shift", torch.tensor(float(self.vae.config.get("shift_factor", 0.0))), persistent=False)
        self.register_buffer("_scale", torch.tensor(float(self.vae.config.get("scaling_factor", 1.0))), persistent=False)

        if dtype == "bf16":
            self.vae = self.vae.to(dtype=torch.bfloat16)
            self.amp_dtype = torch.bfloat16
        elif dtype == "fp16":
            self.vae = self.vae.to(dtype=torch.float16)
            self.amp_dtype = torch.float16
        else:
            self.vae = self.vae.to(dtype=torch.float32)
            self.amp_dtype = None

        self.vae = self.vae.to(device).eval()
        if freeze:
            for p in self.vae.parameters():
                p.requires_grad_(False)

        self._enforce_fp32 = bool(enforce_fp32)

    @property
    def latent_channels(self) -> int:
        return int(getattr(self.vae.config, "latent_channels", 16))

    @property
    def scaling_factor(self) -> float:
        return float(self._scale.item())

    @property
    def shift_factor(self) -> float:
        return float(self._shift.item())

    def _maybe_autocast(self):
        # If we enforce fp32, disable autocast
        if self._enforce_fp32 or self.amp_dtype is None:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.amp_dtype)

    def _ensure_range(self, x: torch.Tensor) -> torch.Tensor:
        # Accept [0,1] or [-1,1]; map to [-1,1]
        if x.min() >= 0.0 and x.max() <= 1.0:
            x = x * 2.0 - 1.0
        return x

    @torch.no_grad()
    def encode(self, images: torch.Tensor, sample_posterior: bool = True) -> torch.Tensor:
        x = images.to(device=self.vae.device, dtype=self.vae.dtype)
        x = self._ensure_range(x)

        with self._maybe_autocast():
            posterior = self.vae.encode(x).latent_dist
            z = posterior.sample() if sample_posterior else posterior.mean
            z_scaled = (z - self._shift) * self._scale
        return z_scaled

    @torch.no_grad()
    def decode(self, z_scaled: torch.Tensor) -> torch.Tensor:
        z_scaled = z_scaled.to(device=self.vae.device, dtype=self.vae.dtype)

        with self._maybe_autocast():
            z = z_scaled / self._scale + self._shift
            x_rec = self.vae.decode(z).sample
        return x_rec

    @torch.no_grad()
    def reconstruct(self, images: torch.Tensor, sample_posterior: bool = False) -> torch.Tensor:
        z = self.encode(images, sample_posterior=sample_posterior)
        return self.decode(z)


if __name__ == "__main__":
    vae = FluxVAE(pretrained_path="/PATH/TO/flux/ckpt/vae").eval().cuda()
    x = torch.randn(1,3,256,256, device="cuda").tanh()  # any image; random is fine here

    with torch.no_grad():
        z_scaled = vae.encode(x, sample_posterior=False)  # (z - shift) * scale
        sf = (z_scaled.flatten().std() + 1e-12).reciprocal()  # 1/std
        # Two decodes that should match:
        rec1 = vae.decode(z_scaled)
        rec2 = vae.decode((sf * z_scaled) / sf)

    mae = (rec1 - rec2).abs().mean().item()
    print("MAE(rec1, rec2) =", mae) 