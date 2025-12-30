import os
import torch
from torch.utils import data
import numpy as np
import csv
import random
import hashlib, json
from pathlib import Path
from PIL import Image
import re

class ObjaverseLVIS(data.Dataset):
    """
    Raw Data Loader for Objaverse-LVIS (.npz files).
    """
    def __init__(
        self,
        split,
        transform=None,
        conditioning_type='uncond',
        # Sampling Params
        sdf_sampling=True,
        num_vol_samples=1024,
        num_near_samples=1024,
        surface_sampling=True,
        surface_size=8192,
        dataset_folder='/path/to/your/data',
        return_sdf=True, 
        verbose=False,
        max_dataset_size=None,
        train_csv_path: str | None = None,
        val_csv_path: str | None = None,
        **kwargs,
    ):
        self.surface_size, self.transform = surface_size, transform
        self.sdf_sampling, self.num_vol_samples, self.num_near_samples = sdf_sampling, num_vol_samples, num_near_samples
        self.split, self.conditioning_type = split, conditioning_type
        self.surface_sampling, self.data_folder = surface_sampling, Path(dataset_folder)
        self.return_sdf, self.verbose = return_sdf, verbose

        allowed = {'uncond', 'text', 'gobjaverse', 'lvis'}
        if self.conditioning_type not in allowed:
            raise ValueError(f"conditioning_type must be one of {allowed}")

        # --- Load Manifest ---
        if split == 'train':
            csv_path = Path(train_csv_path or 'utils/objaverse_lvis_train.csv')
        else:
            csv_path = Path(val_csv_path or f'utils/objaverse_lvis_{split}.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"LVIS CSV not found at {csv_path}")

        # --- Build LVIS Map ---
        if self.conditioning_type == 'lvis':
            with open(csv_path, newline='', encoding='utf-8') as f:
                rdr = csv.DictReader(f)
                cats = sorted({(row.get('lvis_category') or '').strip() for row in rdr if row.get('lvis_category')})
            self.lvis_category_to_id = {cat: i for i, cat in enumerate(cats)}

        # --- Parse Rows ---
        self.models = []
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                uid = (row.get('model_uid') or '').strip()
                if not uid: continue
                
                npz_path = self.data_folder / f"{uid}.npz"
                if not npz_path.exists():
                    if self.verbose: print(f"[SKIP] missing file: {npz_path}")
                    continue

                self.models.append({
                    'path': str(npz_path),
                    'model_uid': uid,
                    'lvis_category': (row.get('lvis_category') or '').strip(),
                    'caption': (row.get('cap3d_caption') or 'a 3d model').strip(),
                    'gobjaverse_class_id': int(row.get('class_id') or -1)
                })

        # --- Subsetting ---
        if max_dataset_size is not None and self.split == 'train':
            self.models = self.models[:max_dataset_size]

        print(f"Loaded {len(self.models)} Objaverse-LVIS samples ({split})")

    def __len__(self):
        return len(self.models)

    def __getitem__(self, idx):
        model_info = self.models[idx]
        npz_path = model_info['path']

        try:
            with np.load(npz_path) as data:
                vol_points_full = data.get('vol_points')
                vol_sdf_full = data.get('vol_sdf')
                near_points_full = data.get('near_points')
                near_sdf_full = data.get('near_sdf')
                surface_full = data.get('surface_points')
        except Exception as e:
            print(f"[ERROR] Failed to load {npz_path}: {e}. Retrying...")
            return self.__getitem__(np.random.randint(len(self)))

        # --- Surface Sampling ---
        if self.surface_sampling and surface_full is not None:
            num_avail = surface_full.shape[0]
            sel = np.random.default_rng().choice(num_avail, min(num_avail, self.surface_size), replace=False)
            surface = torch.from_numpy(surface_full[sel]).float()
        else:
            surface = torch.empty(0, 3)

        # --- SDF Sampling (Balanced) ---
        if self.sdf_sampling and vol_points_full is not None:
            pos_mask = vol_sdf_full < 0
            neg_mask = ~pos_mask
            n_pos = int(self.num_vol_samples // 2)
            n_neg = self.num_vol_samples - n_pos

            n_pos_avail, n_neg_avail = pos_mask.sum(), neg_mask.sum()
            
            # Logic to balance positive/negative samples or fallback
            if n_pos_avail > 0:
                sel = np.random.default_rng().choice(n_pos_avail, n_pos, replace=n_pos_avail < n_pos)
                pos_pts, pos_sdf = vol_points_full[pos_mask][sel], vol_sdf_full[pos_mask][sel]
            else:
                sel = np.random.default_rng().choice(n_neg_avail, n_pos, replace=True)
                pos_pts, pos_sdf = vol_points_full[neg_mask][sel], vol_sdf_full[neg_mask][sel]

            if n_neg_avail > 0:
                sel = np.random.default_rng().choice(n_neg_avail, n_neg, replace=n_neg_avail < n_neg)
                neg_pts, neg_sdf = vol_points_full[neg_mask][sel], vol_sdf_full[neg_mask][sel]
            else:
                sel = np.random.default_rng().choice(n_pos_avail, n_neg, replace=True)
                neg_pts, neg_sdf = vol_points_full[pos_mask][sel], vol_sdf_full[pos_mask][sel]

            vol_points = np.concatenate([pos_pts, neg_pts], axis=0)
            vol_sdf = np.concatenate([pos_sdf, neg_sdf], axis=0)

            # Near surface
            n_near_avail = near_points_full.shape[0]
            sel_near = np.random.default_rng().choice(n_near_avail, self.num_near_samples, replace=n_near_avail < self.num_near_samples)
            
            points = torch.cat([torch.from_numpy(vol_points), torch.from_numpy(near_points_full[sel_near])], dim=0).float()
            sdf = torch.cat([torch.from_numpy(vol_sdf), torch.from_numpy(near_sdf_full[sel_near])], dim=0).float()
        else:
            points, sdf = torch.empty(0, 3), torch.empty(0)

        # --- Transform ---
        if self.transform:
            surface, points = self.transform(surface, points)
        elif self.split == 'train':
            # Default lightweight aug; this is for VecsetX training, since our diffusion only trained on latent, we didn't modify this part.
            perm = torch.randperm(3)
            if points.numel(): points = points[:, perm]
            if surface.numel(): surface = surface[:, perm]
            sign = (torch.randint(2, (3,)) * 2 - 1)
            if points.numel(): points *= sign[None]
            if surface.numel(): surface *= sign[None]

        labels = sdf if self.return_sdf else (sdf < 0).float()

        # --- Conditioning ---
        if self.conditioning_type == 'uncond':
            conditioning = model_info['model_uid']
        elif self.conditioning_type == 'text':
            conditioning = model_info['caption']
        elif self.conditioning_type == 'gobjaverse':
            conditioning = model_info['gobjaverse_class_id']
        elif self.conditioning_type == 'lvis':
            conditioning = self.lvis_category_to_id.get(model_info['lvis_category'], -1)
        else:
            raise ValueError("Invalid conditioning")

        return {
            'points': points, 'labels': labels, 'surface': surface,
            'conditioning': conditioning, 'source_path': npz_path
        }


class LatentDataset(data.Dataset):
    """
    Shape Latent Dataset (Standard 3D VAE latents).
    Assumes pre-computed .pt files.
    """
    def __init__(self, latent_folder, split='train', dataset_type='objaverse_lvis',
                 conditioning_type='uncond', max_dataset_size=None, **kwargs):
        super().__init__()
        self.latent_folder = Path(latent_folder).resolve()
        self.split = split
        self.conditioning_type = conditioning_type
        
        if dataset_type != 'objaverse_lvis':
             raise ValueError(f"LatentDataset only supports 'objaverse_lvis'")

        # --- Discovery ---
        print(f"[{split}] Scanning {self.latent_folder}...")
        all_files = sorted(list(self.latent_folder.rglob("*.pt")))
        if not all_files: raise RuntimeError(f"No .pt files found.")

        # --- Metadata ---
        csv_path = kwargs.get('val_csv_path') if split != 'train' else kwargs.get('train_csv_path')
        if csv_path is None: csv_path = f'utils/objaverse_lvis_{split}.csv'
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        if self.conditioning_type == 'lvis':
            with open(csv_path, newline='', encoding='utf-8') as f:
                rdr = csv.DictReader(f)
                lvis_cats = sorted({(row.get('lvis_category') or '').strip() for row in rdr if row.get('lvis_category')})
            self.lvis_category_to_id = {cat: i for i, cat in enumerate(lvis_cats)}

        self.model_metadata = {}
        with open(csv_path, newline='', encoding='utf-8') as csvfile:
            for row in csv.DictReader(csvfile):
                uid = (row.get('model_uid') or '').strip()
                if uid:
                    self.model_metadata[uid] = {
                        'lvis_category': (row.get('lvis_category') or '').strip(),
                        'gobjaverse_class_id': int(row.get('class_id') or -1),
                        'text': (row.get('cap3d_caption') or 'a 3d model').strip()
                    }

        # --- Grouping & Intersection ---
        self.model_groups = {}
        for file_path in all_files:
            # Assumes filename structure like: {uid}_aug{n}.pt
            rel = str(file_path.relative_to(self.latent_folder)).replace('\\', '/')
            model_uid = rel.rsplit('_aug', 1)[0].rsplit('.pt', 1)[0]
            self.model_groups.setdefault(model_uid, []).append(file_path)

        self.unique_models = sorted(list(self.model_groups.keys() & self.model_metadata.keys()))
        print(f"Matched {len(self.unique_models)} unique models.")

        # --- Deterministic Subsetting to make sure everytime we use same dataset to training (resuming) ---
        self.seed = int(kwargs.get("seed", 0))
        self.snapshot_dir = Path(kwargs.get("snapshot_dir") or "splits")
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        if max_dataset_size is not None and split == 'train' and max_dataset_size < len(self.unique_models):
            snap_key = {"type": "latent_shape", "split": split, "csv": str(csv_path), "max": max_dataset_size, "seed": self.seed}
            snap_hash = hashlib.sha1(json.dumps(snap_key, sort_keys=True).encode()).hexdigest()[:12]
            snap_file = self.snapshot_dir / f"latentuids_{split}_{max_dataset_size}_{snap_hash}.txt"

            if snap_file.exists():
                keep = [l.strip() for l in open(snap_file)]
                self.unique_models = [u for u in keep if u in set(self.unique_models)][:max_dataset_size]
            else:
                rng = np.random.default_rng(self.seed)
                idx = rng.permutation(len(self.unique_models))[:max_dataset_size]
                self.unique_models = [self.unique_models[i] for i in idx]
                with open(snap_file, 'w') as f: f.write('\n'.join(self.unique_models))

    def __len__(self):
        return len(self.unique_models)

    def __getitem__(self, idx):
        model_uid = self.unique_models[idx]
        # Randomly select an augmented view
        latent_path = random.choice(self.model_groups[model_uid])
        
        try:
            latents = torch.load(latent_path, weights_only=True)
            meta = self.model_metadata[model_uid]
            
            cond = 0
            if self.conditioning_type == 'text':
                cond = meta['text']
            elif self.conditioning_type == 'lvis':
                cond = self.lvis_category_to_id.get(meta.get('lvis_category', ""), -1)
            elif self.conditioning_type == 'gobjaverse':
                cond = meta['gobjaverse_class_id']

            return {'latents': latents, 'conditioning': cond}

        except Exception as e:
            print(f"[Error] {model_uid}: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))
    
class LatentImageDataset(data.Dataset):
    """
    Image Latent Dataset (e.g., FluxVAE latents).
    """
    def __init__(
        self,
        latent_folder,
        split="train",
        expect_hw: int = 32, expect_c: int = 16, flatten_to_tokens: bool = True,
        dataset_type: str = "objaverse_lvis", conditioning_type: str = "uncond",
        max_dataset_size: int | None = None,
        train_csv_path: str | None = None, val_csv_path: str | None = None,
        train_view_policy: str | None = None, val_view_policy: str = "center",
        seed: int = 0, **kwargs,
    ):
        super().__init__()
        if dataset_type != "objaverse_lvis": raise ValueError("Only Objaverse-LVIS supported.")
        if conditioning_type == "class": conditioning_type = "lvis"
        
        self.split, self.latent_folder = split, Path(latent_folder).resolve()
        self.expect_hw, self.expect_c, self.flatten = int(expect_hw), int(expect_c), bool(flatten_to_tokens)
        self.conditioning_type, self.seed = conditioning_type, int(seed)
        self.train_view_policy, self.val_view_policy = train_view_policy, val_view_policy

        # --- Discovery ---
        self.files = sorted(self.latent_folder.rglob("*.pt"))
        if not self.files: raise RuntimeError(f"No .pt files found.")

        csv_path = Path(train_csv_path or "utils/objaverse_lvis_train.csv") if split == "train" else Path(val_csv_path or f"utils/objaverse_lvis_{split}.csv")
        
        if self.conditioning_type == 'lvis':
            with open(csv_path, newline="", encoding="utf-8") as f:
                rdr = csv.DictReader(f)
                cats = sorted({(row.get("lvis_category") or "").strip() for row in rdr if row.get("lvis_category")})
            self.lvis_category_to_id = {cat: i for i, cat in enumerate(cats)}

        self.model_metadata = {}
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                uid = (row.get("model_uid") or "").strip()
                if uid:
                    self.model_metadata[uid] = {
                        "lvis_category": (row.get("lvis_category") or "").strip(),
                        "text": (row.get("caption") or "a 3d object").strip()
                    }

        # --- Grouping ---
        groups = {}
        for p in self.files:
            # Strip view suffix like _view00 to find base model
            stem = str(p.relative_to(self.latent_folder)).replace("\\", "/").rsplit(".pt", 1)[0]
            base = re.sub(r"(?:_view\d+|_aug\d+)$", "", stem)
            groups.setdefault(base, []).append(p)

        self.model_groups = {u: groups[u] for u in groups.keys() & self.model_metadata.keys()}
        self.unique_models = sorted(self.model_groups.keys())

        # --- Subsetting ---
        self.snapshot_dir = Path(kwargs.get("snapshot_dir") or "images_splits")
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        if max_dataset_size is not None and self.split == "train" and max_dataset_size < len(self.unique_models):
            snap_key = {"type": "latent_img", "split": split, "csv": str(csv_path), "max": max_dataset_size, "seed": self.seed}
            snap_hash = hashlib.sha1(json.dumps(snap_key, sort_keys=True).encode()).hexdigest()[:12]
            snap_file = self.snapshot_dir / f"latentuids_img_{split}_{max_dataset_size}_{snap_hash}.txt"

            if snap_file.exists():
                keep = [l.strip() for l in open(snap_file)]
                self.unique_models = [u for u in keep if u in set(self.unique_models)][:max_dataset_size]
            else:
                rng = np.random.default_rng(self.seed)
                idx = rng.permutation(len(self.unique_models))[:max_dataset_size]
                self.unique_models = [self.unique_models[i] for i in idx]
                with open(snap_file, "w", encoding="utf-8") as f:
                    f.write('\n'.join(self.unique_models))

        print(f"[LatentImageDataset] {len(self.unique_models)} unique models.")

    def __len__(self):
        return len(self.unique_models)

    def _to_tokens(self, z: torch.Tensor) -> torch.Tensor:
        """Reshape (C, H, W) -> (T, C)"""
        if z.ndim == 3:
            C, H, W = z.shape
            if (H, W) != (self.expect_hw, self.expect_hw) or C != self.expect_c:
                raise ValueError(f"Shape mismatch: {z.shape}")
            return z.view(C, H * W).permute(1, 0).contiguous()
        return z

    def _select_view_idx(self, policy: str, n: int) -> int:
        if policy == "first": return 0
        if policy == "center": return n // 2
        if policy == "random": return random.randint(0, n - 1)
        if isinstance(policy, str) and policy.startswith("index:"):
            try: return max(0, min(n - 1, int(policy.split(":", 1)[1])))
            except: return 0
        return 0

    def __getitem__(self, idx: int):
        uid = self.unique_models[idx]
        group = self.model_groups.get(uid, [])
        if not group: return self.__getitem__(np.random.randint(0, len(self)))

        # Select view based on policy (random for train, deterministic for val)
        policy = self.train_view_policy if self.split == "train" else self.val_view_policy
        if not policy and self.split == "train": policy = "random"
        
        j = self._select_view_idx(policy, len(group))
        fp = group[j]

        try:
            obj = torch.load(fp, map_location="cpu", weights_only=True)
            # Handle dict vs raw tensor format
            z = obj if torch.is_tensor(obj) else (obj.get("latents") or obj.get("z"))
            
            if self.flatten: z = self._to_tokens(z)

            meta = self.model_metadata[uid]
            cond = 0
            if self.conditioning_type == "text": 
                cond = meta["text"]
            elif self.conditioning_type == "lvis":
                cond = self.lvis_category_to_id.get(meta.get("lvis_category") or "", -1)

            # Return 'view_idx' to track which view was actually used
            return {"latents": z, "conditioning": cond, "source_path": str(fp), "view_idx": int(j)}

        except Exception as e:
            print(f"[LatentImageDataset] failed on {fp}: {e}")
            return self.__getitem__(np.random.randint(0, len(self)))