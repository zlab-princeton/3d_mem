import os
import re
from typing import Dict, List, Union, Optional

import torch
import numpy as np
import trimesh
from PIL import Image
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only

from hy3dshape.pipelines import export_to_trimesh


def _safe_name(s: str, limit: int = 60) -> str:
    """Sanitize a string to be a valid filename."""
    s = str(s)
    # If string looks like a PIL object repr, use a generic name
    if "PIL.Image" in s or "Tensor" in s:
        return "sample"
    return re.sub(r'[^a-zA-Z0-9_\-\.]+', '_', s)[:limit]

def _as_pil_image(obj: object) -> Optional[Image.Image]:
    """Convert tensor/numpy array to PIL Image."""
    if isinstance(obj, Image.Image):
        return obj
    
    # Handle Tensor (C, H, W) or (H, W, C)
    if torch.is_tensor(obj):
        x = obj.detach().cpu()
        if x.ndim == 3:
            if x.shape[0] in (1, 3, 4): x = x.permute(1, 2, 0) 
            x = x.float()
            # Normalize [-1, 1] -> [0, 1]
            if x.min() < 0.0: x = (x.clamp(-1, 1) + 1.0) * 0.5
            x = (x.clamp(0, 1) * 255.0).byte().numpy()
            return Image.fromarray(x)
            
    # Handle Numpy
    if isinstance(obj, np.ndarray):
        x = obj
        if x.ndim == 3:
            if x.shape[0] in (1, 3, 4): x = np.moveaxis(x, 0, -1)
            if x.dtype.kind == 'f':
                if x.min() < 0.0: x = (np.clip(x, -1, 1) + 1.0) * 0.5
                x = (np.clip(x, 0, 1) * 255.0 + 0.5).astype(np.uint8)
            return Image.fromarray(x)
    return None

def _save_condition_image(visual_dir: str, base_name: str, condition) -> None:
    """If the condition is an image, save a PNG preview alongside the mesh."""
    img = _as_pil_image(condition)
    if img is not None:
        os.makedirs(visual_dir, exist_ok=True)
        cond_png = os.path.join(visual_dir, f"{base_name}_cond.png")
        img.thumbnail((256, 256))
        img.save(cond_png)


class ConditionalMeshLogger(Callback):
    """
    Logs generated 3D meshes to local disk as .glb files.
    """
    def __init__(
        self,
        step_frequency: int,
        num_samples: int = 1,
        bounds: Union[List[float], Tuple[float]] = (-1.01, -1.01, -1.01, 1.01, 1.01, 1.01),
        **kwargs
    ) -> None:
        super().__init__()
        self.step_freq = step_frequency
        self.num_samples = int(num_samples)

        # Geometry/sampling kwargs passed to pipeline
        octree_resolution = kwargs.pop("octree_resolution", None)
        octree_depth = kwargs.pop("octree_depth", None)
        if octree_resolution is None and octree_depth is not None:
            octree_resolution = 1 << int(octree_depth)

        self.sample_kwargs = {
            "output_type": "latents2mesh",
            "bounds": bounds,
            "octree_resolution": octree_resolution,
            "mc_level": kwargs.pop("mc_level", 0.0),
            "num_chunks": kwargs.pop("num_chunks", 20000),
            "enable_pbar": False,
        }

    @rank_zero_only
    def _log_one(self, mesh_like, visual_dir: str, condition, index: int) -> None:
        tm = export_to_trimesh(mesh_like)
        if not isinstance(tm, trimesh.Trimesh):
            return

        # Generate Filename
        safe_cond = _safe_name(condition)
        base_name = f"{index:02d}_{safe_cond}"

        # Save Conditioning Image (if applicable)
        _save_condition_image(visual_dir, base_name, condition)

        # Save GLB
        out_glb = os.path.join(visual_dir, f"{base_name}.glb")
        os.makedirs(os.path.dirname(out_glb), exist_ok=True)
        tm.export(out_glb)

    @rank_zero_only
    def log_local(
        self,
        outputs: List[List],
        conditions: List,
        save_dir: str, split: str,
        global_step: int, current_epoch: int, batch_idx: int,
    ) -> None:
        folder = f"gs-{global_step:010d}_e-{current_epoch:06d}_b-{batch_idx:06d}"
        visual_dir = os.path.join(save_dir, "visuals", split, folder)
        
        # Extract meshes (outputs is a list of lists, usually [[mesh1, mesh2...]])
        batch_meshes = outputs[0] if len(outputs) > 0 else []
        
        n = min(len(batch_meshes), len(conditions))
        for i in range(n):
            if batch_meshes[i] is None: continue
            self._log_one(batch_meshes[i], visual_dir, condition=conditions[i], index=i)

    @rank_zero_only
    def log_sample(self, pl_module: pl.LightningModule, batch: Dict, batch_idx: int, split: str = "train") -> None:
        was_train = pl_module.training
        if was_train: pl_module.eval()
        
        run_dir = getattr(pl_module.logger, "log_dir", None) or pl_module.logger.save_dir

        try:
            # Extract subset of conditions
            if isinstance(batch['conditioning'], (list, tuple)):
                cond_list = list(batch['conditioning'])[:self.num_samples]
            elif torch.is_tensor(batch['conditioning']):
                cond_list = batch['conditioning'][:self.num_samples].detach().cpu().tolist()
            else:
                cond_list = [batch['conditioning']]

            # Run Inference
            with torch.no_grad():
                outputs = pl_module.sample(
                    conditioning=cond_list,
                    batch_size=len(cond_list),
                    **self.sample_kwargs
                )

            self.log_local(
                outputs=outputs,
                conditions=cond_list,
                save_dir=run_dir,
                split=split,
                global_step=pl_module.global_step,
                current_epoch=pl_module.current_epoch,
                batch_idx=batch_idx
            )
        except Exception as e:
            print(f"[MeshLogger] Sampling failed: {e}")
        finally:
            if was_train: pl_module.train()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if pl_module.global_step > 0 and pl_module.global_step % self.step_freq == 0:
            self.log_sample(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx == 0:
            self.log_sample(pl_module, batch, batch_idx, split="val")