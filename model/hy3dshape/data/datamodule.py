import pytorch_lightning as pl
from torch.utils.data import DataLoader, get_worker_info
from pathlib import Path
import numpy as np
import random
import torch
from PIL import Image
from torch.utils.data._utils.collate import default_collate

# Only import LVIS-related datasets (Cleaned version)
from .dataset import LatentDataset, LatentImageDataset, ObjaverseLVIS

def _collate_keep_pil(batch):
    """Custom collate function to preserve PIL Images (used for image conditioning)."""
    out = {}
    keys = batch[0].keys()
    for k in keys:
        vals = [b[k] for b in batch]
        if k == "conditioning" and isinstance(vals[0], Image.Image):
            out[k] = vals  # Return list[PIL.Image] instead of stacking tensors
        else:
            out[k] = default_collate(vals)
    return out

def _worker_init_fn(worker_id: int):
    """Ensures each data loader worker has a different random seed."""
    base_seed = torch.initial_seed() % 2**32
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)
    torch.manual_seed(base_seed + worker_id)
    info = get_worker_info()
    if info is not None and hasattr(info.dataset, "set_rng"):
        try:
            info.dataset.set_rng(np.random.default_rng(base_seed + worker_id))
        except Exception:
            pass

class UnifiedDataModule(pl.LightningDataModule):
    """
    Main DataModule for 3D generation.
    Supports modes:
    1. 'raw': Loads .npz files (Point clouds, SDFs) for VAE training.
    2. 'latent': Loads .pt files (encoded latents) for DiT training.
       - Supports both Shape Latents and Image Latents via `latent_is_image` flag.
    """
    def __init__(
        self,
        mode: str,
        batch_size: int,
        num_workers: int = 4,
        # --- Paths for Latent Mode ---
        train_latent_folder: str | None = None,
        val_latent_folder: str | None = None,
        # --- Paths for Raw Mode ---
        dataset_folder: str | None = None,
        val_dataset_folder: str | None = None,
        # --- Configs ---
        dataset_type: str = 'objaverse_lvis', 
        train_csv_path: str | None = None,
        val_csv_path: str | None = None,
        # --- Sampling Params (Raw Mode) ---
        surface_size: int = 8192,
        conditioning_type: str = 'uncond',
        sdf_sampling: bool = False,
        num_vol_samples: int = 1024,
        num_near_samples: int = 1024,
        surface_sampling: bool = True,
        # --- Advanced Params ---
        max_dataset_size: int | None = None,
        prefetch_factor: int = 4,
        return_sdf: bool = False,
        snapshot_dir: str | None = None,
        latent_is_image: bool = False,       # True = Image Latents, False = Shape Latents
        expect_hw: int = 32, expect_c: int = 16,
        flatten_to_tokens: bool = True,
        train_view_policy: str | None = None,
        val_view_policy: str = "center", 
    ):
        super().__init__()
        self.save_hyperparameters()

    def _build_raw_dataset(self, split: str):
        """Helper to instantiate the raw ObjaverseLVIS dataset."""
        # Fallback: Use train folder for val if val folder is missing
        root = self.hparams.dataset_folder if split == 'train' else (
            self.hparams.val_dataset_folder or self.hparams.dataset_folder
        )
        common = dict(
            split=split,
            dataset_folder=root,
            conditioning_type=self.hparams.conditioning_type,
            surface_sampling=self.hparams.surface_sampling,
            sdf_sampling=self.hparams.sdf_sampling,
            num_vol_samples=self.hparams.num_vol_samples,
            num_near_samples=self.hparams.num_near_samples,
            surface_size=self.hparams.surface_size,
            return_sdf=self.hparams.return_sdf,
            verbose=False,
            # Limit dataset size only for training to speed up debugging if needed
            max_dataset_size=self.hparams.max_dataset_size if split == 'train' else None,
        )
        
        if self.hparams.dataset_type.lower() == 'objaverse_lvis':
            return ObjaverseLVIS(
                train_csv_path=self.hparams.train_csv_path,
                val_csv_path=self.hparams.val_csv_path,
                **common
            )
        else:
            raise ValueError(f"UnifiedDataModule only supports 'objaverse_lvis', got: {self.hparams.dataset_type}")

    def setup(self, stage: str | None = None):
        """Initializes train/val datasets based on mode."""
        if self.hparams.mode == 'latent':
            val_folder = self.hparams.val_latent_folder or self.hparams.train_latent_folder
            
            # --- Branch A: Image Latents (e.g., from FluxVAE) ---
            if getattr(self.hparams, "latent_is_image", False):
                print(f"Setting up LatentImageDataset (View Policy: {self.hparams.train_view_policy})")
                common = dict(
                    dataset_type=self.hparams.dataset_type,
                    conditioning_type=self.hparams.conditioning_type,
                    train_csv_path=self.hparams.train_csv_path,
                    val_csv_path=self.hparams.val_csv_path,
                    expect_hw=self.hparams.expect_hw,
                    expect_c=self.hparams.expect_c,
                    flatten_to_tokens=self.hparams.flatten_to_tokens,
                    train_view_policy=self.hparams.train_view_policy,
                    val_view_policy=self.hparams.val_view_policy,
                    snapshot_dir=self.hparams.snapshot_dir,
                )
                self.train_dataset = LatentImageDataset(
                    latent_folder=self.hparams.train_latent_folder, split='train', **common,
                    max_dataset_size=self.hparams.max_dataset_size,
                )
                self.val_dataset = LatentImageDataset(
                    latent_folder=val_folder, split='val', **common
                )
            
            # --- Branch B: Shape Latents (Standard 3D VAE) ---
            else:
                _train_args = dict(
                    latent_folder=self.hparams.train_latent_folder,
                    split='train',
                    conditioning_type=self.hparams.conditioning_type,
                    max_dataset_size=self.hparams.max_dataset_size,
                    dataset_type=self.hparams.dataset_type,
                    train_csv_path=self.hparams.train_csv_path,
                    val_csv_path=self.hparams.val_csv_path,
                )
                if self.hparams.snapshot_dir:
                    _train_args["snapshot_dir"] = self.hparams.snapshot_dir
                
                self.train_dataset = LatentDataset(**_train_args, batch_size_hint=self.hparams.batch_size,)
                self.val_dataset = LatentDataset(
                    latent_folder=val_folder,
                    split='val',
                    conditioning_type=self.hparams.conditioning_type,
                    dataset_type=self.hparams.dataset_type,
                    train_csv_path=self.hparams.train_csv_path,
                    val_csv_path=self.hparams.val_csv_path,
                    batch_size_hint=self.hparams.batch_size,
                )
        
        # --- Branch C: Raw Data (for VAE Training) ---
        elif self.hparams.mode == 'raw':
            self.train_dataset = self._build_raw_dataset('train')
            self.val_dataset   = self._build_raw_dataset('val')
        else:
            raise ValueError(f"Unknown mode: {self.hparams.mode}")

        # Safety Check
        if len(self.train_dataset) == 0:
            raise RuntimeError("Train dataset is empty after setup() — check paths/CSVs.")
        if len(self.val_dataset) == 0:
            raise RuntimeError("Val dataset is empty after setup() — check paths/CSVs.")

    def train_dataloader(self):
        # Disable persistent workers if passing heavy PIL images to avoid pickling overhead
        is_image_cond = (self.hparams.conditioning_type == 'image')
        eff_prefetch = 2 if is_image_cond else self.hparams.prefetch_factor
        eff_persist  = False if is_image_cond else (self.hparams.num_workers > 0)

        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=eff_persist,
            prefetch_factor=(eff_prefetch if self.hparams.num_workers > 0 else None),
            worker_init_fn=_worker_init_fn,
            collate_fn=_collate_keep_pil if is_image_cond else None,
        )

    def val_dataloader(self):
        is_image_cond = (self.hparams.conditioning_type == 'image')
        eff_prefetch = 2 if is_image_cond else self.hparams.prefetch_factor
        eff_persist  = False if is_image_cond else (self.hparams.num_workers > 0)
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=eff_persist,
            prefetch_factor=(eff_prefetch if self.hparams.num_workers > 0 else None),
            worker_init_fn=_worker_init_fn,
            collate_fn=_collate_keep_pil if is_image_cond else None,
        )