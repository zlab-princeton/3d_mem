import os
import torch
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only
from typing import Dict, List


def _to_01(x: torch.Tensor) -> torch.Tensor:
    """Map images to [0,1] for saving/logging. Accepts BCHW or CHW."""
    if x.dtype.is_floating_point:
        # if any negative values, assume [-1,1] and rescale
        if torch.min(x) < 0:
            x = (x.clamp(-1, 1) + 1) * 0.5
        return x.clamp(0, 1)
    # uint8 -> float [0,1]
    if x.dtype == torch.uint8:
        return (x.float() / 255.0).clamp(0, 1)
    return x.float().clamp(0, 1)

class ImageLogger(Callback):
    def __init__(
        self,
        step_frequency: int,
        num_samples: int = 4,
        guidance_scale: float = 5.0,
        save_images: bool = True,
        out_dirname: str = "visuals_img",
        nrow: int = 4,
        **kwargs,  # absorb extra YAML keys
    ) -> None:
        super().__init__()
        self.step_freq = int(step_frequency)
        self.num_samples = int(num_samples)
        self.guidance_scale = float(guidance_scale)
        self.save_images = bool(save_images)
        self.out_dirname = str(out_dirname)
        self.nrow = int(nrow)

        # toggled by on_train_batch_end, consumed at next validation batch
        self.log_next_val_batch = False

    def _root_dir(self, pl_module: pl.LightningModule) -> str:
        # TensorBoardLogger has .log_dir; fall back to .save_dir as needed
        logger = getattr(pl_module, "logger", None)
        return getattr(logger, "log_dir", None) or getattr(logger, "save_dir", ".")

    @rank_zero_only
    def _save_and_log(self, pl_module: pl.LightningModule, images: torch.Tensor, split: str) -> None:
        """
        images: BCHW in [0,1] float
        """
        if images.ndim == 3:
            images = images.unsqueeze(0)

        grid = torchvision.utils.make_grid(images, nrow=min(self.nrow, images.size(0)))
        tag = f"{split}/generated_images"
        tb = getattr(pl_module.logger, "experiment", None)
        if tb is not None and hasattr(tb, "add_image"):
            tb.add_image(tag, grid, global_step=pl_module.global_step)

        if not self.save_images:
            return

        run_dir = self._root_dir(pl_module)
        folder = f"gs-{pl_module.global_step:010d}_e-{pl_module.current_epoch:06d}"
        out_dir = os.path.join(run_dir, self.out_dirname, split, folder)
        os.makedirs(out_dir, exist_ok=True)

        grid_path = os.path.join(out_dir, "grid.png")
        torchvision.utils.save_image(grid, grid_path)

        for i in range(min(images.size(0), self.num_samples)):
            img = images[i]
            img_path = os.path.join(out_dir, f"sample_{i:02d}.png")
            torchvision.utils.save_image(img, img_path)
            

    @rank_zero_only
    def log_sample_images(self, pl_module: pl.LightningModule, batch: Dict, split: str) -> None:
        was_train = pl_module.training
        if was_train:
            pl_module.eval()

        try:
            cond_list: List = []
            if "conditioning" in batch and self.num_samples > 0:
                conditions = batch["conditioning"]
                if isinstance(conditions, (list, tuple)):
                    cond_list = list(conditions)[: self.num_samples]
                elif torch.is_tensor(conditions):
                    cond_list = conditions[: self.num_samples].detach().cpu().tolist()

            if not cond_list:
                cond_list = [None] * self.num_samples  # unconditional or missing

            with torch.no_grad():
                out = pl_module.sample(
                    conditioning=cond_list,
                    batch_size=len(cond_list),
                    guidance_scale=self.guidance_scale,
                    output_type="tensor",   # expects BCHW in either [-1,1] or [0,1]
                )
                images = out[0] if isinstance(out, list) else out

            if not torch.is_tensor(images):
                return
            images = _to_01(images.detach().cpu())  # -> [0,1] float
            self._save_and_log(pl_module, images, split)
        finally:
            if was_train:
                pl_module.train()

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx: int) -> None:
        if pl_module.global_step > 0 and (pl_module.global_step % self.step_freq == 0):
            self.log_next_val_batch = True

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if batch_idx == 0 and self.log_next_val_batch:
            self.log_sample_images(pl_module, batch, split="val")
            self.log_next_val_batch = False