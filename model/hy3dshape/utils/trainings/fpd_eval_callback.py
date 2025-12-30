from __future__ import annotations
import os, io, gc, csv, json, math, traceback, hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
from contextlib import nullcontext

import numpy as np
import torch
import pytorch_lightning as pl

from hy3dshape.utils.metrics.fpd import compute_mu_sigma, frechet_distance
from hy3dshape.utils.metrics.encoders import build_encoder

try:
    import point_cloud_utils as pcu
    _HAS_PCU = True
except Exception:
    _HAS_PCU = False
    import trimesh as _tm


def _mesh_to_points(tri_verts: np.ndarray, tri_faces: np.ndarray, n_points: int, normalize_first: bool = False) -> np.ndarray:
    V = tri_verts.astype(np.float64, copy=False)
    F = tri_faces.astype(np.int32, copy=False)

    if normalize_first:
        c = (V.max(axis=0) + V.min(axis=0)) * 0.5
        V = V - c
        r = np.linalg.norm(V, axis=1).max()
        V = V / max(float(r), 1e-8)

    if _HAS_PCU:
        fid, bc = pcu.sample_mesh_random(V, F, n_points)
        pts = pcu.interpolate_barycentric_coords(F, fid, bc, V)
    else:
        tm = _tm.Trimesh(vertices=V.astype(np.float32), faces=F, process=False)
        pts, _ = tm.sample(n_points, return_index=True)
        pts = pts.astype(np.float64)

    if not normalize_first:
        c = (pts.max(axis=0) + pts.min(axis=0)) * 0.5
        pts = pts - c
        r = np.linalg.norm(pts, axis=1).max()
        pts = pts / max(float(r), 1e-8)

    return pts.astype(np.float32)


@dataclass
class FPDConfig:
    # reference/test set
    test_csv: str
    cache_dir: str
    ref_npz_root: str               # dir containing <uid>.npz for ref clouds
    ref_npz_key: str = "surface_points"
    reference_sample_points: int = 4096

    # NEW: multi-encoder support
    encoders: Optional[List[Dict[str, Any]]] = None
    # Single encoder
    encoder: Optional[Dict[str, Any]] = None

    
    pointnet_ckpt: str = ""
    width_mult: int = 2
    device_batch_size: int = 64
    normal_channel: bool = False

    # eval sizing
    n_eval: int = 500
    n_points: int = 4096

    # conditioning
    condition_type: str = "lvis"   # "lvis"/"class"/"text"/"image"/"uncond"
    lvis_map_json: Optional[str] = None

    # schedule
    heavy_every_n_steps: int = 10000
    also_run_on_validation: bool = False

    # sampling: meshing kwargs (identical to mesh_logger)
    sampler: Dict[str, Any] = None
    
    distributed: bool = True


def _ddp_info(trainer: pl.Trainer) -> Tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def _build_lvis_map_from_csv(csv_path: str) -> Dict[str, int]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        cats = sorted({(r.get("lvis_category") or "").strip()
                       for r in csv.DictReader(f) if r.get("lvis_category")})
    return {c: i for i, c in enumerate(cats)}


def _read_rows(csv_path: str, n: int) -> List[Dict[str, str]]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rows.append(r)
            if len(rows) >= n:
                break
    return rows


def _normalize_unit_sphere(v: np.ndarray) -> np.ndarray:
    c = (v.max(axis=0) + v.min(axis=0)) * 0.5
    v = v - c
    r = np.linalg.norm(v, axis=1).max()
    r = max(float(r), 1e-6)
    return v / r


def _list_shard(lst: Sequence, rank: int, world_size: int) -> List:
    if world_size <= 1:
        return list(lst)
    return [x for i, x in enumerate(lst) if (i % world_size) == rank]


def _all_gather_features_object(local_feats: np.ndarray, device: torch.device) -> np.ndarray:
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return local_feats.astype(np.float32, copy=False)
    obj = local_feats.astype(np.float32, copy=False)
    gathered: List[Optional[np.ndarray]] = [None for _ in range(torch.distributed.get_world_size())]
    with torch.inference_mode(False):
        torch.distributed.all_gather_object(gathered, obj)
    sizes = [0 if g is None else g.shape[0] for g in gathered]
    if sum(sizes) == 0:
        return np.zeros((0, obj.shape[1] if obj.ndim == 2 else 0), dtype=np.float32)
    return np.concatenate([g for g in gathered if g is not None and g.size > 0], axis=0)


class ModelEvalFPDCallback(pl.Callback):
    """
    Generate once per eval, then compute FPD for one or many encoders on the same sample set.
    Encoders are pluggable via hy3dshape.utils.metrics.encoders.build_encoder.
    """
    def __init__(self, **kwargs):
        self.cfg = FPDConfig(**kwargs)
        os.makedirs(self.cfg.cache_dir, exist_ok=True)
        self._last_ran_step = -1
        self._encoders = None  # type: Optional[List[Any]]  # built lazily

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if self.cfg.heavy_every_n_steps > 0 and (step % self.cfg.heavy_every_n_steps == 0) and step > 0:
            self._maybe_run_eval(trainer, pl_module, tag=f"train_step={step}")

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.cfg.also_run_on_validation:
            self._maybe_run_eval(trainer, pl_module, tag="val_epoch_end")

    def _maybe_run_eval(self, trainer: pl.Trainer, pl_module: pl.LightningModule, tag: str):
        rank, world_size = _ddp_info(trainer)
        if trainer.global_step == self._last_ran_step:
            return
        self._last_ran_step = trainer.global_step

        try:
            self._run_eval_distributed(trainer, pl_module, rank, world_size, tag)
        except Exception as e:
            if rank == 0:
                print("[FPD] Evaluation failed:", repr(e))
                traceback.print_exc()

    @torch.no_grad()
    def _run_eval_distributed(self, trainer: pl.Trainer, pl_module: pl.LightningModule, rank: int, world_size: int, tag: str):
        device = pl_module.device
        dist = self.cfg.distributed and world_size > 1 and torch.distributed.is_initialized()
        error = torch.tensor([0], device=device) if dist else None
        try:
            # LVIS/class/text/image mapping/conds
            if self.cfg.lvis_map_json and os.path.isfile(self.cfg.lvis_map_json):
                lvis_map = json.load(open(self.cfg.lvis_map_json, "r"))
            else:
                lvis_map = _build_lvis_map_from_csv(self.cfg.test_csv)

            rows_all = _read_rows(self.cfg.test_csv, self.cfg.n_eval)
            ctype = (self.cfg.condition_type or "lvis").lower()

            if ctype in ("lvis", "class"):
                if self.cfg.lvis_map_json and os.path.isfile(self.cfg.lvis_map_json):
                    lvis_map = json.load(open(self.cfg.lvis_map_json, "r"))
                else:
                    lvis_map = _build_lvis_map_from_csv(self.cfg.test_csv)
                pairs = [(r["model_uid"], (r.get("lvis_category") or "").strip()) for r in rows_all]
                pairs = [(uid, cat) for uid, cat in pairs if cat in lvis_map]
                uids_all = [uid for uid, _ in pairs]
                cond_all = [int(lvis_map[cat]) for _, cat in pairs]
            elif ctype == "text":
                uids_all = [r["model_uid"] for r in rows_all]
                cond_all = [(r.get("cap3d_caption") or "a 3d object").strip() for r in rows_all]
            elif ctype == "image":
                from PIL import Image
                def _crop_random_tile(view_path: str, V: int = 12):
                    if not view_path or not os.path.isfile(view_path): return None
                    img = Image.open(view_path).convert("RGB"); W, H = img.size
                    if H % V != 0: return img
                    h = H // V; i = np.random.randint(0, V); top = i * h
                    return img.crop((0, top, W, top + h))
                uids_all = [r["model_uid"] for r in rows_all]
                cond_all = [(_crop_random_tile(r.get("view_path") or "") or Image.new("RGB", (256, 256), (127, 127, 127)))
                            for r in rows_all]
            elif ctype == "uncond":
                uids_all = [r["model_uid"] for r in rows_all]
                cond_all = [0] * len(rows_all)
            else:
                raise ValueError(f"Unknown condition_type: {self.cfg.condition_type}")

            rows_shard = _list_shard(list(zip(uids_all, cond_all)), rank, world_size)

            # Sampler kwargs; translate octree depth if needed
            samp_kwargs = dict(self.cfg.sampler or {})
            if "octree_depth" in samp_kwargs and "octree_resolution" not in samp_kwargs:
                samp_kwargs["octree_resolution"] = 2 ** int(samp_kwargs.pop("octree_depth"))

            # Init encoders (once per rank)
            if self._encoders is None:
                self._encoders = self._build_encoders(device)

            # Generate once → point clouds
            local_pcs: List[np.ndarray] = []
            per_gpu_batch = 32
            rng = torch.Generator(device=device).manual_seed(1234 + rank)

            pl_module.eval()
            was_train = pl_module.training
            prec = str(trainer.precision)
            use_amp = prec not in ("32", "32-true", "32-precision")
            amp_dtype = torch.bfloat16 if prec.startswith("bf16") else torch.float16
            ema_ctx = getattr(pl_module, "ema_scope", None)
            ema_ctx = ema_ctx if callable(ema_ctx) else (lambda *_args, **_kwargs: nullcontext())

            try:
                with ema_ctx("FPD-eval"), torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    print(f"[LOG rank{rank}] Evaluating FPD (generate once)...")
                    for i0 in range(0, len(rows_shard), per_gpu_batch):
                        chunk = rows_shard[i0:i0 + per_gpu_batch]
                        if not chunk:
                            break
                        _, conds = zip(*chunk) if len(chunk) > 0 else ([], [])
                        outs = pl_module.sample(
                            conditioning=list(conds),
                            batch_size=len(conds),
                            generator=rng,
                            output_type='trimesh',
                            **samp_kwargs
                        )
                        outs = outs[0] if isinstance(outs, list) else outs  # BC
                        for m in outs:
                            if m is None:
                                continue
                            tri_verts, tri_faces = self._export_to_numpy(m)
                            pts = _mesh_to_points(tri_verts, tri_faces, getattr(self, "_sample_n_points", self.cfg.n_points))
                            pts = _normalize_unit_sphere(pts)
                            local_pcs.append(pts)
            finally:
                pl_module.train(was_train)
                torch.cuda.empty_cache()

            # For each encoder: encode locally, gather, load/build ref stats, compute FPD
            for enc in self._encoders:
                if len(local_pcs) == 0:
                    print(f"[FPD][{tag}] WARNING: no generated point clouds; skipping {getattr(enc,'name','enc')}.")
                    continue

                local_feats = self._encode_pcs_with_encoder(enc, local_pcs)
                all_feats = _all_gather_features_object(local_feats, device) if (self.cfg.distributed and world_size > 1 and torch.distributed.is_initialized()) else local_feats

                if rank == 0:
                    mu_r, sig_r, n_ref, ref_path = self._load_or_build_ref_stats(uids_all, enc)
                    mu_g, sig_g = compute_mu_sigma(all_feats)
                    fpd = frechet_distance(mu_g, sig_g, mu_r, sig_r)
                    tag_name = f"metrics/fpd_{getattr(enc, 'name', 'enc')}"
                    pl_module.log(tag_name, float(fpd), on_step=False, on_epoch=True, logger=True, prog_bar=True, rank_zero_only=True)
                    print(f"[FPD][{tag}] ref_n={n_ref} gen_n={all_feats.shape[0]}  {tag_name}={float(fpd):.3f}  (stats: {ref_path})")

            if self.cfg.distributed and world_size > 1 and torch.distributed.is_initialized():
                try:
                    trainer.strategy.barrier()
                except Exception:
                    torch.distributed.barrier(device_ids=[device.index] if device.type == "cuda" else None)

            del local_pcs
            gc.collect()
        except Exception as e:
            if rank == 0:
                print("[FPD] Evaluation failed:", repr(e))
                traceback.print_exc()
            if dist:
                error.fill_(1)
        finally:
            if dist:
                torch.distributed.all_reduce(error, op=torch.distributed.ReduceOp.SUM)
                trainer.strategy.barrier()
                if error.item() > 0:
                    return

    def _export_to_numpy(self, mesh_obj) -> Tuple[np.ndarray, np.ndarray]:
        try:
            import trimesh as _tm
            if isinstance(mesh_obj, _tm.Trimesh):
                V = np.asarray(mesh_obj.vertices, dtype=np.float32)
                F = np.asarray(mesh_obj.faces, dtype=np.int32)
                return V, F
        except Exception:
            pass
        V, F = mesh_obj
        V = np.asarray(V, dtype=np.float32)
        F = np.asarray(F, dtype=np.int32)
        return V, F

    def _ref_cache_paths(self, encoder, uids_all: List[str]) -> Tuple[str, str, str]:
        cache_tag = getattr(encoder, "cache_tag", "unknown-encoder")
        uids_fpr = hashlib.md5(",".join(uids_all).encode("utf-8")).hexdigest()[:10]
        ref_key = (
            f"{self.cfg.test_csv}|{self.cfg.ref_npz_root}|{self.cfg.ref_npz_key}|"
            f"{self.cfg.reference_sample_points}|{self.cfg.n_points}|{uids_fpr}"
        )
        ref_hash = hashlib.md5(ref_key.encode()).hexdigest()[:10]
        stats_path = os.path.join(self.cfg.cache_dir, f"ref_stats_{cache_tag}_{ref_hash}.npz")
        old_embed_path = os.path.join(self.cfg.cache_dir, f"ref_embed_{cache_tag}_{ref_hash}.npy")
        meta_path = os.path.join(self.cfg.cache_dir, f"ref_stats_{cache_tag}_{ref_hash}.meta.json")
        return stats_path, meta_path, old_embed_path

    def _load_or_build_ref_stats(self, uids_all: List[str], encoder) -> Tuple[np.ndarray, np.ndarray, int, str]:
        """
        Returns (mu, sigma, n, stats_path). Builds and caches if missing.
        """
        stats_path, meta_path, old_embed_path = self._ref_cache_paths(encoder, uids_all)
        if os.path.isfile(stats_path):
            z = np.load(stats_path)
            mu_r = z["mu"]; sig_r = z["sigma"]; n_ref = int(z["n"])
            return mu_r, sig_r, n_ref, stats_path

        if os.path.isfile(old_embed_path):
            print(f"[FPD] Converting legacy embedding cache → stats: {old_embed_path}")
            feats = np.load(old_embed_path)
            mu_r, sig_r = compute_mu_sigma(feats)
            np.savez_compressed(stats_path, mu=mu_r, sigma=sig_r, n=feats.shape[0], feature_dim=feats.shape[1])
            try:
                meta = {
                    "encoder": getattr(encoder, "cache_tag", "enc"),
                    "feature_dim": int(feats.shape[1]),
                    "n_points": int(self.cfg.n_points),
                    "reference_sample_points": int(self.cfg.reference_sample_points),
                    "csv": self.cfg.test_csv,
                    "ref_npz_root": self.cfg.ref_npz_root,
                    "ref_npz_key": self.cfg.ref_npz_key,
                }
                with open(meta_path, "w") as f:
                    json.dump(meta, f)
            except Exception:
                pass
            return mu_r, sig_r, int(feats.shape[0]), stats_path
        
        print(f"[FPD] Building reference stats → {stats_path}")
        rng = np.random.RandomState(0)
        batch_pts, feats_all = [], []
        bsz = int(getattr(encoder, "device_batch_size", self.cfg.device_batch_size))
        for uid in uids_all:
            npz_path = os.path.join(self.cfg.ref_npz_root, f"{uid}.npz")
            obj = np.load(npz_path)
            surf = obj[self.cfg.ref_npz_key]
            nref = surf.shape[0]
            k = self.cfg.reference_sample_points
            replace = nref < k
            idx = rng.choice(nref, k if not replace else min(k, nref), replace=replace)
            pts = _normalize_unit_sphere(surf[idx])
            batch_pts.append(pts)
            if len(batch_pts) >= bsz:
                feats_all.append(encoder.encode_np(np.stack(batch_pts, 0)))
                batch_pts = []
        if batch_pts:
            feats_all.append(encoder.encode_np(np.stack(batch_pts, 0)))

        feats = np.concatenate(feats_all, axis=0) if feats_all else np.zeros((0, getattr(encoder, "feature_dim", 512)), dtype=np.float32)
        mu_r, sig_r = compute_mu_sigma(feats)
        np.savez_compressed(stats_path, mu=mu_r, sigma=sig_r, n=feats.shape[0], feature_dim=feats.shape[1])
        try:
            meta = {
                "encoder": getattr(encoder, "cache_tag", "enc"),
                "feature_dim": int(feats.shape[1]),
                "n_points": int(self.cfg.n_points),
                "reference_sample_points": int(self.cfg.reference_sample_points),
                "csv": self.cfg.test_csv,
                "ref_npz_root": self.cfg.ref_npz_root,
                "ref_npz_key": self.cfg.ref_npz_key,
            }
            with open(meta_path, "w") as f:
                json.dump(meta, f)
        except Exception:
            pass
        return mu_r, sig_r, int(feats.shape[0]), stats_path

    def _build_encoders(self, device):
        # precedence: encoders > encoder > legacy PointNet
        enc_cfgs: List[Dict[str, Any]] = []
        if self.cfg.encoders:
            enc_cfgs = [dict(ec) for ec in self.cfg.encoders]   # shallow copy
        elif self.cfg.encoder:
            enc_cfgs = [dict(self.cfg.encoder)]
        else:
            enc_cfgs = [dict(
                name="pointnet",
                ckpt=self.cfg.pointnet_ckpt,
                width_mult=self.cfg.width_mult,
                device_batch_size=self.cfg.device_batch_size,
                normal_channel=self.cfg.normal_channel,
            )]

        # provide a default n_points for each encoder (can be overridden per-encoder in YAML)
        default_n = int(getattr(self.cfg, "n_points", 4096))
        for ec in enc_cfgs:
            ec.setdefault("n_points", default_n)

        encoders = [build_encoder(ec, device=device) for ec in enc_cfgs]

        for ecfg, enc in zip(enc_cfgs, encoders):
            if not hasattr(enc, "n_points"):
                try:
                    setattr(enc, "n_points", int(ecfg.get("n_points", default_n)))
                except Exception:
                    setattr(enc, "n_points", default_n)

        self._sample_n_points = max([default_n] + [int(getattr(e, "n_points", default_n)) for e in encoders])

        try:
            names = [getattr(e, "name", type(e).__name__) for e in encoders]
            per_n = [int(getattr(e, "n_points", default_n)) for e in encoders]
            print(f"[FPD] encoders={names}  n_points={per_n}  sample_n_points={self._sample_n_points}")
        except Exception:
            pass

        return encoders

    def _encode_pcs_with_encoder(self, encoder, pcs_list):
        target_n = int(getattr(encoder, "n_points", len(pcs_list[0])))
        rng = np.random.RandomState(42)  # deterministic subsampling
        bsz = int(getattr(encoder, "device_batch_size", self.cfg.device_batch_size))
        out, curN = [], pcs_list[0].shape[0]
        for i in range(0, len(pcs_list), bsz):
            batch = pcs_list[i:i+bsz]
            if curN != target_n:
                sel = rng.choice(curN, target_n, replace=False)
                batch = [b[sel] for b in batch]
            feats = encoder.encode_np(np.stack(batch, 0).astype(np.float32))
            out.append(feats)
        return np.concatenate(out, 0) if out else np.zeros((0, getattr(encoder, "feature_dim", 512)), dtype=np.float32)