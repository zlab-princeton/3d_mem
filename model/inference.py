import os, csv, json, argparse, time, subprocess, sys, tempfile, random
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import torch
from omegaconf import OmegaConf
from PIL import Image

from hy3dshape.utils import get_config_from_file, instantiate_from_config

# utils

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def flatten_outputs(outputs):
    if outputs is None:
        return []
    if isinstance(outputs, list) and len(outputs) == 1 and isinstance(outputs[0], list):
        return outputs[0]
    if isinstance(outputs, list):
        return outputs
    return [outputs]

def save_trimesh_item(obj, path: Path) -> bool:
    try:
        import numpy as np
        import trimesh
    except ImportError:
        print("[WARN] trimesh not installed; skipping mesh export.")
        return False

    if path.suffix.lower() != ".glb":
        path = path.with_suffix(".glb")

    def to_scene(x) -> Optional["trimesh.Scene"]:
        if x is None:
            return None
        if isinstance(x, trimesh.Scene):
            return x
        if isinstance(x, trimesh.Trimesh):
            return trimesh.Scene(x)
        if isinstance(x, (list, tuple)):
            geoms = []
            for it in x:
                if isinstance(it, trimesh.Scene):
                    geoms.extend(it.geometry.values())
                elif isinstance(it, trimesh.Trimesh):
                    geoms.append(it)
                elif isinstance(it, dict) and {"vertices", "faces"} <= set(it.keys()):
                    try:
                        geoms.append(
                            trimesh.Trimesh(
                                vertices=np.asarray(it["vertices"]),
                                faces=np.asarray(it["faces"]),
                                process=False,
                            )
                        )
                    except Exception:
                        pass
            if not geoms:
                return None
            return trimesh.Scene(geoms)
        if isinstance(x, dict) and {"vertices", "faces"} <= set(x.keys()):
            try:
                m = trimesh.Trimesh(
                    vertices=np.asarray(x["vertices"]),
                    faces=np.asarray(x["faces"]),
                    process=False,
                )
                return trimesh.Scene(m)
            except Exception:
                return None
        for meth in ("to_trimesh", "as_trimesh"):
            if hasattr(x, meth):
                try:
                    m = getattr(x, meth)()
                    if isinstance(m, trimesh.Trimesh):
                        return trimesh.Scene(m)
                except Exception:
                    pass
        return None

    scene = to_scene(obj)
    if scene is None:
        return False

    try:
        scene.export(str(path), file_type="glb", include_normals=True)
        return True
    except Exception as e:
        print(f"[WARN] GLB export failed for {path.name}: {e}")
        try:
            m = scene.dump(concatenate=True) if hasattr(scene, "dump") else None
            if m is None:
                geoms = list(scene.geometry.values())
                if geoms:
                    import trimesh
                    m = trimesh.util.concatenate(geoms) if len(geoms) > 1 else geoms[0]
            if m is not None and hasattr(m, "vertices") and hasattr(m, "faces"):
                np.savez_compressed(path.with_suffix(".npz"), vertices=m.vertices, faces=m.faces)
                print(f"[save-fallback] wrote vertices/faces -> {path.with_suffix('.npz')}")
                return True
        except Exception as e2:
            print(f"[WARN] fallback NPZ failed for {path.name}: {e2}")
        return False

def deduce_raw_cond_type(model) -> str:
    c = getattr(model, "cond_stage_model", None)
    t = getattr(c, "type", None)
    if t in ("uncond", "class", "text", "image", "lvis", "gobjaverse"):
        return t
    t2 = getattr(model, "cond_type", None)
    if t2 in ("uncond", "class", "text", "image", "lvis", "gobjaverse"):
        return t2
    return "uncond"

def resolve_final_cond_type(cfg: OmegaConf, raw_cond_type: str) -> str:
    # "class" is a meta type; dataset decides (lvis/gobjaverse)
    if raw_cond_type == "class":
        ds = getattr(cfg, "dataset", None)
        params = getattr(ds, "params", {}) if ds is not None else {}
        dataset_cond = params.get("conditioning_type", "uncond")
        if dataset_cond not in ("lvis", "gobjaverse"):
            raise ValueError(
                f"When conditioner type is 'class', dataset.conditioning_type must be 'lvis' or 'gobjaverse', got '{dataset_cond}'"
            )
        return dataset_cond
    return raw_cond_type

def _open_manifest(path: Path, fieldnames: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    f = open(path, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=fieldnames)
    if not exists:
        w.writeheader()
    return f, w

# Checkpoint resolution / merging

def _is_ds_dir(p: Path) -> bool:
    if not p.is_dir():
        return False
    if (p / "latest").is_file() and (p / "checkpoint").is_dir():
        return True
    if any(x.name.startswith("mp_rank_") for x in p.iterdir() if x.is_file()):
        return True
    if p.name == "checkpoint":
        return True
    if (p / "checkpoint").exists():
        return True
    return False

def _find_ds_tag_dir(ds_root: Path) -> Optional[Path]:
    if (ds_root / "latest").is_file() and (ds_root / "checkpoint").is_dir():
        tag = (ds_root / "latest").read_text().strip()
        cand = ds_root / "checkpoint" / tag
        if cand.is_dir():
            return cand
    ck = ds_root / "checkpoint" if (ds_root / "checkpoint").is_dir() else ds_root
    tags = [d for d in ck.iterdir() if d.is_dir() and d.name.startswith("global_step")]
    if tags:
        return sorted(tags)[-1]
    if any((ck / x).is_file() for x in os.listdir(ck) if x.startswith("mp_rank_")):
        return ck
    return None

def _canonicalize_ds_for_merge(p: Path) -> Path:
    p = p.resolve()
    if (p / "latest").is_file() and (p / "checkpoint").is_dir():
        return p
    if p.name == "checkpoint" and (p.parent / "latest").is_file():
        return p.parent
    tag = _find_ds_tag_dir(p)
    return tag or p

def _discover_merged_file(out_dir: Path) -> Path:
    for name in ["pytorch_model_fp32.bin", "pytorch_model.bin", "pytorch_model.pt"]:
        p = out_dir / name
        if p.is_file():
            return p
    cands = list(out_dir.glob("*.bin")) + list(out_dir.glob("*.pt")) + list(out_dir.glob("*.safetensors"))
    if cands:
        return max(cands, key=lambda x: x.stat().st_size)
    raise RuntimeError(f"DeepSpeed merge produced no model file in {out_dir}")

def _merge_deepspeed_checkpoint(ds_ckpt_dir: Path, out_dir: Path) -> Path:
    ds_ckpt_dir = _canonicalize_ds_for_merge(ds_ckpt_dir)
    out_dir = ensure_dir(Path(out_dir))
    cmds = []
    for candidate in [ds_ckpt_dir, ds_ckpt_dir.parent]:
        z2 = candidate / "zero_to_fp32.py"
        if z2.exists():
            cmds.append([sys.executable, str(z2), str(ds_ckpt_dir), str(out_dir)])
    cmds.append([sys.executable, "-m", "deepspeed.utils.zero_to_fp32", str(ds_ckpt_dir), str(out_dir)])

    last_err = None
    for cmd in cmds:
        print(f"[deepspeed] merge: {' '.join(cmd)}")
        try:
            res = subprocess.run(cmd, cwd=str(ds_ckpt_dir), capture_output=True, text=True, check=True)
            if res.stdout.strip():
                print(res.stdout.strip())
            merged_file = _discover_merged_file(out_dir)
            print(f"[deepspeed] merged -> {merged_file}")
            return merged_file
        except subprocess.CalledProcessError as e:
            last_err = e
            print(f"[deepspeed] merge failed (trying next):\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}")
        except Exception as e:
            last_err = e
            print(f"[deepspeed] merge output handling failed (trying next): {e}")
    raise RuntimeError(f"DeepSpeed merge failed for {ds_ckpt_dir}. Last error: {last_err}")

def _try_load_ds_rank0_module(ds_any: Path) -> Optional[Dict[str, Any]]:
    root = ds_any
    if root.name == "checkpoint" and root.parent.exists():
        root = root.parent
    tag = _find_ds_tag_dir(root) or (ds_any if any(ds_any.glob("mp_rank_*")) else None)
    if tag is None:
        return None
    cand = tag / "mp_rank_00_model_states.pt"
    if not cand.exists():
        return None
    obj = torch.load(cand, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "module" in obj and isinstance(obj["module"], dict):
        print("[deepspeed] fallback: using mp_rank_00_model_states.pt['module']")
        return obj["module"]
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        print("[deepspeed] fallback: using mp_rank_00_model_states.pt['state_dict']")
        return obj["state_dict"]
    return None

def _resolve_ckpt_target(path_like: str) -> Tuple[Path, str]:
    p = Path(path_like)
    if p.is_file():
        if p.name.endswith("pytorch_model.bin.index.json"):
            return p, "hf_index"
        return p, "file"
    if p.is_dir():
        cks = sorted(p.glob("*.ckpt"))
        if cks:
            return cks[-1], "file"
        idx = p / "pytorch_model.bin.index.json"
        if idx.exists():
            return idx, "hf_index"
        merged = p / "pytorch_model_fp32.bin"
        if merged.exists():
            return merged, "file"
        if _is_ds_dir(p):
            return p, "ds_dir"
    raise FileNotFoundError(f"Could not resolve a checkpoint from: {path_like}")

def _load_ckpt_object(path: Path) -> Dict[str, torch.Tensor] | Dict[str, Any]:
    if path.name.endswith("pytorch_model.bin.index.json"):
        with open(path, "r", encoding="utf-8") as f:
            index = json.load(f)
        folder = path.parent
        state: Dict[str, torch.Tensor] = {}
        for part in sorted(set(index.get("weight_map", {}).values())):
            state.update(torch.load(folder / part, map_location="cpu", weights_only=False))
        return state
    return torch.load(path, map_location="cpu", weights_only=False)

def _extract_state_dict(obj: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    if isinstance(obj, dict) and "module" in obj and isinstance(obj["module"], dict):
        return obj["module"]
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        return obj["model"]
    if isinstance(obj, dict):
        return obj
    raise ValueError("Unsupported checkpoint object format")

# ---- parsing / proof helpers ----

def _strip_fwd(k: str) -> str:
    return k.replace("_forward_module.", "")

def _debug_topk(sd: Dict[str, torch.Tensor], k: int = 10, label: str = "state_dict"):
    keys = sorted(sd.keys())
    print(f"  total entries in {label}: {len(keys)}")
    print(f"  top-{k} keys (sorted):")
    for name in keys[:k]:
        v = sd[name]
        shp = tuple(getattr(v, "shape", ())) if hasattr(v, "shape") else "?"
        dt = str(getattr(v, "dtype", "?"))
        print(f"    - {name}: shape={shp} dtype={dt}")

def _debug_counts(sd: Dict[str, torch.Tensor]):
    def has_pref(pref):
        return sum(1 for k in sd if _strip_fwd(k).startswith(pref))
    print(f"[debug] key counts: model.*={has_pref('model.')}  "
          f"cond_stage_model.*={has_pref('cond_stage_model.')}  "
          f"first_stage_model.*={has_pref('first_stage_model.')}  total={len(sd)}")

def _split_blocks(sd: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    den, cond = {}, {}
    for k, v in sd.items():
        k2 = _strip_fwd(k)
        if k2.startswith("module."): k2 = k2[7:]
        if k2.startswith("model."):
            den[k2[len("model."):]] = v
        elif k2.startswith("cond_stage_model."):
            cond[k2[len("cond_stage_model."):]] = v
    return den, cond

# z-scale recovery / override / std

def _extract_z_scale_value(sd: Dict[str, Any]) -> Optional[float]:
    for k in ("z_scale_factor", "module.z_scale_factor", "_forward_module.z_scale_factor"):
        if k in sd:
            v = sd[k]
            if isinstance(v, torch.Tensor):
                try:
                    return float(v.detach().flatten()[0].cpu().item())
                except Exception:
                    pass
            try:
                return float(v)
            except Exception:
                pass
    return None

def _try_load_z_scale_from_ckpt(ckpt_arg: str) -> Optional[float]:
    try:
        resolved, _ = _resolve_ckpt_target(ckpt_arg)
        obj = _load_ckpt_object(resolved)
        if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
            z = _extract_z_scale_value(obj["state_dict"])
            if z is not None:
                print(f"[z-scale] found in Lightning state_dict: {z:.6f}")
                return z
        sd = _extract_state_dict(obj)
        z = _extract_z_scale_value(sd)
        if z is not None:
            print(f"[z-scale] found in raw state: {z:.6f}")
            return z
    except Exception as e:
        print(f"[z-scale] probe failed: {e}")
    return None

def _get_vae_scale_if_any(ae) -> Optional[float]:
    if ae is None:
        return None
    cand = ["z_scale_factor", "scaling_factor", "scale_factor", "latent_scale", "latent_scaling"]
    for name in cand:
        if hasattr(ae, name):
            v = getattr(ae, name)
            if isinstance(v, torch.Tensor):
                try:
                    return float(v.detach().flatten()[0].cpu().item())
                except Exception:
                    continue
            try:
                return float(v)
            except Exception:
                continue
    cfg = getattr(ae, "config", None)
    if cfg is not None:
        for name in ["scaling_factor", "scale_factor", "latent_scale", "z_scale_factor"]:
            v = getattr(cfg, name, None)
            if v is not None:
                try:
                    return float(v)
                except Exception:
                    pass
    return None

def _apply_z_scale_to_model(model, z: float):
    val = float(z)
    if hasattr(model, "z_scale_factor"):
        cur = getattr(model, "z_scale_factor")
        if isinstance(cur, torch.Tensor):
            with torch.no_grad():
                cur.data.fill_((val))
        else:
            setattr(model, "z_scale_factor", val)
    else:
        setattr(model, "z_scale_factor", val)
    print(f"[z-scale] using z_scale_factor={val:.6f}")

    pipe = getattr(model, "pipeline", None)
    if pipe is not None:
        if hasattr(pipe, "set_latent_scale"):
            try:
                pipe.set_latent_scale(val)
                print("[z-scale] forwarded to pipeline via set_latent_scale()")
            except Exception:
                pass
        elif hasattr(pipe, "z_scale_factor"):
            try:
                setattr(pipe, "z_scale_factor", val)
                print("[z-scale] set pipeline.z_scale_factor")
            except Exception:
                pass

# ---- STD-based estimation helpers ----

def _maybe_to_tensor(x):
    if torch.is_tensor(x):
        return x
    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
    except Exception:
        pass
    return torch.as_tensor(x)

def _load_calib_latents(path: Path) -> Optional[torch.Tensor]:
    try:
        obj = torch.load(path, map_location="cpu")
        if torch.is_tensor(obj):
            return obj
        if isinstance(obj, dict):
            for k in ("latents", "z", "z_q"):
                if k in obj and torch.is_tensor(obj[k]):
                    return obj[k]
        try:
            import numpy as np
            arr = np.load(path)
            if isinstance(arr, np.lib.npyio.NpzFile):
                for k in ("latents", "z", "z_q"):
                    if k in arr.files:
                        return torch.from_numpy(arr[k])
                return torch.from_numpy(arr[arr.files[0]])
            else:
                return torch.from_numpy(arr)
        except Exception:
            pass
    except Exception as e:
        print(f"[z-scale] failed loading latents from {path}: {e}")
    return None

def _iter_surface_tensors_from_dir(root: Path, take: int) -> List[torch.Tensor]:
    paths = []
    for ext in ("*.pt", "*.pth", "*.npy", "*.npz"):
        paths.extend(sorted(root.glob(ext)))
    tensors: List[torch.Tensor] = []
    for p in paths:
        if len(tensors) >= take:
            break
        try:
            if p.suffix in (".pt", ".pth"):
                obj = torch.load(p, map_location="cpu")
                if torch.is_tensor(obj):
                    tensors.append(obj)
                elif isinstance(obj, dict):
                    x = obj.get("surface", obj.get("latents", None))
                    if torch.is_tensor(x):
                        tensors.append(x)
            else:
                import numpy as np
                arr = np.load(p)
                if isinstance(arr, np.lib.npyio.NpzFile):
                    x = arr.get("surface", arr.get("latents", None))
                    if x is None:
                        x = arr[arr.files[0]]
                else:
                    x = arr
                tensors.append(torch.from_numpy(x))
        except Exception as e:
            print(f"[z-scale] skip {p.name}: {e}")
    return tensors

def _load_surfaces_from_csv(rows: List[Dict[str, Any]], max_n: int) -> List[torch.Tensor]:
    cand_cols = ("surface", "surface_path", "path", "data_path")
    paths = []
    for r in rows:
        for c in cand_cols:
            if c in r and r[c]:
                paths.append(r[c])
                break
        if len(paths) >= max_n:
            break
    tensors: List[torch.Tensor] = []
    for p in paths:
        try:
            pp = Path(p)
            if pp.suffix in (".pt", ".pth"):
                obj = torch.load(pp, map_location="cpu")
                if torch.is_tensor(obj):
                    tensors.append(obj)
                elif isinstance(obj, dict):
                    x = obj.get("surface", obj.get("latents", None))
                    if torch.is_tensor(x):
                        tensors.append(x)
            else:
                import numpy as np
                arr = np.load(pp)
                if isinstance(arr, np.lib.npyio.NpzFile):
                    x = arr.get("surface", arr.get("latents", None))
                    if x is None:
                        x = arr[arr.files[0]]
                else:
                    x = arr
                tensors.append(torch.from_numpy(x))
        except Exception as e:
            print(f"[z-scale] skip {p}: {e}")
    return tensors

def _auto_z_from_std(model, device, latents: Optional[torch.Tensor] = None,
                     surfaces: Optional[List[torch.Tensor]] = None) -> Optional[float]:
    with torch.no_grad():
        if latents is None and (not surfaces):
            return None
        if latents is not None:
            z_q = latents.to(device, non_blocking=True)
        else:
            x_list = [ _maybe_to_tensor(x) for x in surfaces ]
            x = torch.stack(x_list, 0).to(device, non_blocking=True)
            with torch.autocast(device_type=("cuda" if device.type=="cuda" else "cpu"), dtype=torch.bfloat16):
                z_q = model.first_stage_model.encode(surf=x, sample_posterior=True)
        z = z_q.detach()
        std = z.flatten().std()
        std = torch.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0).clamp(min=1e-6)
        z_scale = float((1.0 / std).item())
        _apply_z_scale_to_model(model, z_scale)
        print(f"[z-scale] computed from std over {z.numel()} elements -> {z_scale:.6f}")
        return z_scale

# Conditioner loading policy & ckpt loader

def _should_load_conditioner(final_cond_type: str, mode: str) -> bool:
    if mode == "always":
        return True
    if mode == "never":
        return False
    # auto: only for LVIS class conditioning
    return final_cond_type == "lvis"

def load_only_model_and_conditioner(model, ckpt_arg: str,
                                    load_conditioner: bool = True,
                                    expect_conditioner: bool = True) -> None:
    """
    Load denoiser from ckpt always; load conditioner only if load_conditioner==True.
    """
    resolved, kind = _resolve_ckpt_target(ckpt_arg)
    print(f"[ckpt] resolved '{ckpt_arg}' -> {resolved} ({kind})")

    if kind == "ds_dir":
        with tempfile.TemporaryDirectory() as tdir:
            tdir_path = Path(tdir)
            try:
                merged_file = _merge_deepspeed_checkpoint(resolved, tdir_path)
                obj = _load_ckpt_object(merged_file)
            except Exception as e:
                print(f"[deepspeed] loading merged fp32 failed, trying rank-0 fallback: {e}")
                maybe_sd = _try_load_ds_rank0_module(resolved)
                if maybe_sd is None:
                    raise
                obj = maybe_sd  # already a raw sd
    else:
        obj = _load_ckpt_object(resolved)

    raw_sd = _extract_state_dict(obj)
    _debug_counts(raw_sd)
    _debug_topk(raw_sd, 12, label="raw state_dict")

    den_sd, cond_sd = _split_blocks(raw_sd)

    # ---- load denoiser
    miss, unexp = model.model.load_state_dict(den_sd, strict=False)
    print(f"[load/denoiser] params={len(den_sd)}  missing={len(miss)}  unexpected={len(unexp)}")

    # ---- conditioner policy
    if not load_conditioner:
        print("[load/conditioner] SKIPPED by policy (keeping pretrained CLIP/image encoder from config)")
        return

    if getattr(model, "cond_stage_model", None) is not None:
        if expect_conditioner and not cond_sd:
            raise RuntimeError(
                "Checkpoint had NO 'cond_stage_model.*' weights. "
                "Use a Lightning .ckpt or a DS merge that includes the conditioner."
            )
        if cond_sd:
            cmiss, cunexp = model.cond_stage_model.load_state_dict(cond_sd, strict=False)
            print(f"[load/conditioner] params={len(cond_sd)}  missing={len(cmiss)}  unexpected={len(cunexp)}")
        else:
            print("[load/conditioner] SKIPPED (no cond weights in ckpt)")
    else:
        print("[load/conditioner] Model has no cond_stage_model; skipped")

    emb = getattr(getattr(model, "cond_stage_model", None), "embedder", None)
    if emb is not None and hasattr(emb, "embedding"):
        w = emb.embedding.weight.detach().float().cpu()
        print(f"[cond/stats] embedding shape={tuple(w.shape)} mean={w.mean():.6f} std={w.std():.6f}")

# CSV part

def read_lvis_mapping(json_path: Optional[Path]) -> Optional[Dict[str, int]]:
    if json_path is None:
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def rows_from_csv(csv_path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def build_per_row_plan(rows, cond_type: str, out_dir: Path, output_type: str, lvis_map):
    plan: List[Tuple[int, Any, Path]] = []
    for i, row in enumerate(rows):
        uid = (row.get("model_uid") or f"row{i:07d}").strip()
        if cond_type == "uncond":
            cond_val = None
            target = out_dir / (f"{uid}.glb" if output_type == "trimesh" else f"{uid}.pt")
        elif cond_type == "text":
            cap = (row.get("cap3d_caption") or "a 3d model").strip()
            cond_val = cap
            target = out_dir / (f"{uid}.glb" if output_type == "trimesh" else f"{uid}.pt")
        elif cond_type == "image":
            vpath = (row.get("view_path") or "").strip()
            cond_val = vpath
            target = out_dir / (f"{uid}.glb" if output_type == "trimesh" else f"{uid}.pt")
        elif cond_type == "gobjaverse":
            cid_raw = (row.get("class_id") or "").strip()
            try:
                cid = int(cid_raw) if cid_raw != "" else -1
            except Exception:
                cid = -1
            cond_val = cid
            cls_dir = ensure_dir(out_dir / f"class_{cid}")
            target = cls_dir / (f"{uid}.glb" if output_type == "trimesh" else f"{uid}.pt")
        elif cond_type == "lvis":
            if lvis_map is None:
                raise ValueError("lvis conditioning requires a mapping JSON (category -> id).")
            cat = (row.get("lvis_category") or "").strip()
            lid = lvis_map.get(cat, -1)
            if lid < 0:
                print(f"[WARN] LVIS category '{cat}' not in mapping; using -1")
            cond_val = lid
            cls_dir = ensure_dir(out_dir / f"class_{lid}")
            target = cls_dir / (f"{uid}.glb" if output_type == "trimesh" else f"{uid}.pt")
        else:
            raise RuntimeError(f"Unexpected conditioner type: {cond_type}")
        plan.append((i, cond_val, target))
    return plan

def build_inference_plan(num_samples: int, cond_type: str, out_dir: Path, output_type: str, conditioning_value: Any):
    plan: List[Tuple[int, Any, Path]] = []
    for i in range(num_samples):
        uid = f"sample_{i:07d}"
        if cond_type == "uncond":
            cond_val = None
            target = out_dir / (f"{uid}.glb" if output_type == "trimesh" else f"{uid}.pt")
        elif cond_type == "text":
            cond_val = conditioning_value
            target = out_dir / (f"{uid}.glb" if output_type == "trimesh" else f"{uid}.pt")
        elif cond_type == "image":
            cond_val = conditioning_value  # path to sprite (direct mode)
            target = out_dir / (f"{uid}.glb" if output_type == "trimesh" else f"{uid}.pt")
        elif cond_type in ("gobjaverse", "lvis"):
            cond_val = conditioning_value
            cls_dir = ensure_dir(out_dir / f"class_{conditioning_value}")
            target = cls_dir / (f"{uid}.glb" if output_type == "trimesh" else f"{uid}.pt")
        else:
            raise RuntimeError(f"Unexpected conditioner type: {cond_type}")
        plan.append((i, cond_val, target))
    return plan

def chunk_indices(idx_list: List[int], chunk: int) -> List[List[int]]:
    return [idx_list[i:i+chunk] for i in range(0, len(idx_list), chunk)]

# Image sprite helpers

def _pick_view_from_sprite(png_path: str, num_views: int, pick: str, idx: int, rng: random.Random) -> Tuple[Image.Image, int]:
    im = Image.open(png_path)
    # robust convert-after-crop; but if grayscale, convert anyway
    W, H = im.size
    if num_views <= 0:
        raise ValueError("num_views must be > 0")
    view_h = H // num_views
    if view_h <= 0:
        raise ValueError(f"Sprite too small: H={H}, num_views={num_views}")
    if H % num_views != 0:
        print(f"[image-cond] WARNING: H={H} not divisible by {num_views}; using floor(view_h)={view_h}")

    if pick == "random":
        vidx = rng.randrange(num_views)
    elif pick == "center":
        vidx = num_views // 2
    elif pick == "first":
        vidx = 0
    elif pick == "index":
        vidx = max(0, min(num_views - 1, int(idx)))
    else:
        raise ValueError(f"Unknown image pick mode: {pick}")

    top = vidx * view_h
    bottom = min((vidx + 1) * view_h, H)
    crop = im.crop((0, top, W, bottom)).convert("RGB")
    return crop, vidx

def parse_args():
    p = argparse.ArgumentParser("Inference with robust ckpt loading (denoiser + optional conditioner).")
    p.add_argument("-c", "--config", required=True, type=str, help="YAML with `model:` (Diffuser).")
    p.add_argument("--ckpt", required=True, type=str, help="Checkpoint path (file, HF index, or DeepSpeed dir).")

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--csv", type=str,
                      help="CSV headers: model_uid,lvis_category,gobjaverse_category,cap3d_caption,class_id,view_path,(optional) surface_path")
    mode.add_argument("--num_samples", type=int, help="Direct inference: number of samples")

    p.add_argument("--out_dir", type=str, default="samples")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--output_type", type=str, default="trimesh", choices=["trimesh", "pt"])
    p.add_argument("--overwrite", action="store_true",
                   help="Regenerate outputs even if target files already exist (ignores existence checks).")

    # conditional args
    p.add_argument("--class_id", type=int, help="Class ID for gobjaverse/lvis")
    p.add_argument("--text", type=str, help="Text prompt for text conditioning")
    p.add_argument("--image_path", type=str, help="Sprite path for image conditioning (direct mode)")

    # mapping for CSV lvis
    p.add_argument("--lvis_mapping_json", type=str, default=None,
                   help="JSON mapping LVIS category -> contiguous class ID (CSV mode)")

    # conditioner loading policy
    p.add_argument("--cond_from_ckpt_mode", type=str, default="auto",
                   choices=["auto", "always", "never"],
                   help="auto: load cond from ckpt only for LVIS class cond; always: always load; never: never load.")

    # options
    p.add_argument("--guidance_scale", type=float, default=5., help="CFG scale")
    p.add_argument("--no_ema", action="store_true", help="Disable EMA in inference if available.")
    p.add_argument("--torch_compile", action="store_true", help="torch.compile(model.model) for inference.")
    p.add_argument("--allow_missing_conditioner", action="store_true",
                   help="Do not error if cond weights are missing when loading from ckpt.")

    # manifest logging
    p.add_argument("--manifest", type=str, default=None,
                   help="Append per-sample records to this CSV (header auto-written on create).")

    # z-scale
    p.add_argument("--z_scale_factor", type=float, default=None,
                   help="Override latent scaling (if omitted, try ckpt; then optional STD; then VAE; else default).")
    p.add_argument("--auto_z_from_std", action="store_true",
                   help="Compute z_scale_factor = 1/std(latents) from calibration data (uses --calib_latents or --calib_surface_dir, or CSV surface paths).")
    p.add_argument("--calib_latents", type=str, default=None,
                   help="Path to latents file (.pt/.pth/.npy/.npz) for STD estimation.")
    p.add_argument("--calib_surface_dir", type=str, default=None,
                   help="Directory with a few surface tensors (.pt/.npy/.npz) to encode for STD estimation.")
    p.add_argument("--calib_take", type=int, default=8,
                   help="Number of calibration items to use for STD estimation.")

    # image sprite options
    p.add_argument("--image_views", type=int, default=12, help="Number of vertical views in sprite.")
    p.add_argument("--image_pick", type=str, default="random", choices=["random","center","first","index"],
                   help="How to pick the view from the sprite.")
    p.add_argument("--image_index", type=int, default=0, help="Index when --image_pick=index.")
    return p.parse_args()

def main():
    args = parse_args()

    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    out_dir = ensure_dir(Path(args.out_dir))

    # Load config + model (config provides VAE; we will NOT load VAE from ckpt)
    cfg = get_config_from_file(args.config)
    assert "model" in cfg, "Config must contain a `model:` section."
    model = instantiate_from_config(cfg.model)

    # VAE 
    try:
        ae = model.first_stage_model
        total = sum(p.numel() for p in ae.parameters())
        trainable = sum(p.numel() for p in ae.parameters() if p.requires_grad)
        first_param_name, first_param = next(iter(ae.state_dict().items()))
        print(f"[vae] params={total} trainable={trainable}  sample='{first_param_name}' "
              f"shape={tuple(first_param.shape)} dtype={first_param.dtype}")
    except Exception as e:
        print("[vae] sanity check failed:", e)

    # Conditioner type sanity
    raw_cond_type = deduce_raw_cond_type(model)
    final_cond_type = resolve_final_cond_type(cfg, raw_cond_type)
    print(f"[cond] raw='{raw_cond_type}' -> final='{final_cond_type}'")

    # ckpt loading policy for conditioner
    load_cond_from_ckpt = _should_load_conditioner(final_cond_type, args.cond_from_ckpt_mode)
    if load_cond_from_ckpt:
        print("[policy] conditioner will be loaded from CKPT (class/LVIS or 'always' mode).")
    else:
        print("[policy] conditioner will NOT be loaded from CKPT (keeping pretrained CLIP/image encoder from config).")

    # Load denoiser (+ maybe conditioner) ONLY; never VAE
    try:
        load_only_model_and_conditioner(
            model,
            args.ckpt,
            load_conditioner=load_cond_from_ckpt,
            expect_conditioner=(load_cond_from_ckpt and not args.allow_missing_conditioner),
        )
    except Exception as e:
        print(f"[ERROR] checkpoint load failed: {e}")
        raise

    z_from_cli = args.z_scale_factor
    z_from_ckpt = None if z_from_cli is not None else _try_load_z_scale_from_ckpt(args.ckpt)
    z_from_std = None
    z_from_vae = None

    rows = None
    if args.csv is not None:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        rows = rows_from_csv(csv_path)
        print(f"[csv] loaded {len(rows)} rows")

    if z_from_cli is not None:
        _apply_z_scale_to_model(model, z_from_cli)
    elif z_from_ckpt is not None:
        _apply_z_scale_to_model(model, z_from_ckpt)
    else:
        if args.auto_z_from_std:
            latents = None
            surfaces: List[torch.Tensor] = []

            if args.calib_latents:
                latents = _load_calib_latents(Path(args.calib_latents))

            if latents is None:
                if args.calib_surface_dir:
                    surfaces = _iter_surface_tensors_from_dir(Path(args.calib_surface_dir), take=args.calib_take)
                elif rows is not None:
                    surfaces = _load_surfaces_from_csv(rows, max_n=args.calib_take)

            if latents is not None or surfaces:
                z_from_std = _auto_z_from_std(model, device, latents=latents, surfaces=surfaces)
            else:
                print("[z-scale] STD mode requested but no calibration data found; skipping.")

        if z_from_std is None:
            z_from_vae = _get_vae_scale_if_any(getattr(model, "first_stage_model", None))
            if z_from_vae is not None:
                print(f"[z-scale] not in ckpt/STD; using VAE scaling {z_from_vae:.6f}")
                _apply_z_scale_to_model(model, z_from_vae)
            else:
                z_attr = getattr(model, "z_scale_factor", 1.0)
                z_val = float(z_attr.detach().cpu().item() if isinstance(z_attr, torch.Tensor) and z_attr.numel()==1 else z_attr)
                print(f"[z-scale] not found in ckpt/STD/vae; using model-default {z_val:.6f}")

    model = model.to(device)
    if amp_dtype in (torch.bfloat16, torch.float16):
        for p in model.parameters():
            try:
                p.data = p.data.to(amp_dtype)
            except Exception:
                pass
    model.eval()

    if args.no_ema and hasattr(model, "ema_config") and model.ema_config is not None:
        model.ema_config.ema_inference = False

    if args.torch_compile and hasattr(model, "model"):
        model.model = torch.compile(model.model)
        print("[info] torch.compile() enabled.")


    manifest_f = None
    manifest_w = None
    manifest_fields = [
        "uid", "out_path", "ok",
        "cond_type", "cond_id", "cond_text",
        "source", "csv_row", "seed", "step",
        "batch_size", "output_type", "dt_sec"
    ]
    if getattr(args, "manifest", None):
        manifest_f, manifest_w = _open_manifest(Path(args.manifest), manifest_fields)

    # ---- CSV vs direct ----
    if args.csv is not None:
        lvis_map = None
        if final_cond_type == "lvis":
            if not args.lvis_mapping_json:
                raise ValueError("--lvis_mapping_json is required for 'lvis' conditioning (CSV mode).")
            lvis_map = read_lvis_mapping(Path(args.lvis_mapping_json))
            print(f"[lvis] mapping loaded: {len(lvis_map)} categories")

        plan = build_per_row_plan(rows, final_cond_type, out_dir, args.output_type, lvis_map)
        if args.overwrite:
            pending_indices = list(range(len(plan)))
            done_count = 0
        else:
            pending_indices = [i for i, (_, _, tgt) in enumerate(plan) if not tgt.exists()]
            done_count = len(plan) - len(pending_indices)
        print(f"[resume] {done_count}/{len(plan)} already exist -> {len(pending_indices)} to generate")
    else:
        if final_cond_type in ("gobjaverse", "lvis"):
            if args.class_id is None:
                raise ValueError("--class_id is required for gobjaverse/lvis")
            conditioning_value = args.class_id
        elif final_cond_type == "text":
            if args.text is None:
                raise ValueError("--text is required for text conditioning")
            conditioning_value = args.text
        elif final_cond_type == "image":
            if args.image_path is None:
                raise ValueError("--image_path is required for image conditioning (direct mode)")
            conditioning_value = args.image_path
        elif final_cond_type == "uncond":
            conditioning_value = None
        else:
            raise ValueError(f"Unhandled conditioner: {final_cond_type}")

        print(f"[inference] generating {args.num_samples} samples with conditioning: {conditioning_value}")
        plan = build_inference_plan(args.num_samples, final_cond_type, out_dir, args.output_type, conditioning_value)
        if args.overwrite:
            pending_indices = list(range(len(plan)))
            done_count = 0
        else:
            pending_indices = [i for i, (_, _, tgt) in enumerate(plan) if not tgt.exists()]
            done_count = len(plan) - len(pending_indices)
        print(f"[resume] {done_count}/{len(plan)} already exist -> {len(pending_indices)} to generate")
        lvis_map = None  # unused in direct mode

    if not pending_indices:
        print("[done] Nothing to do. ✅")
        if manifest_f is not None:
            manifest_f.close()
        return

    # Batched sampling
    step = 0
    for batch_ids in chunk_indices(pending_indices, args.batch_size):
        batch_cond_vals: List[Any] = []
        batch_targets: List[Path] = []
        for idx in batch_ids:
            _, cond_val, tgt = plan[idx]
            batch_cond_vals.append(cond_val)
            batch_targets.append(tgt)

        conditioning = None
        picked_views: List[int] = []  # for image logging

        if final_cond_type == "uncond":
            conditioning = None
        elif final_cond_type in ("gobjaverse", "lvis"):
            conditioning = [int(x) for x in batch_cond_vals]
        elif final_cond_type == "text":
            conditioning = batch_cond_vals
        elif final_cond_type == "image":
            images: List[Image.Image] = []
            rng = random.Random(args.seed + step)  # deterministic across runs
            for k, path in enumerate(batch_cond_vals):
                try:
                    img, vidx = _pick_view_from_sprite(
                        path, args.image_views, args.image_pick, args.image_index, rng
                    )
                    images.append(img)
                    picked_views.append(vidx)
                except Exception as e:
                    print(f"[image-cond] ERROR processing '{path}': {e}; using gray placeholder.")
                    images.append(Image.new("RGB", (224, 224), (127, 127, 127)))
                    picked_views.append(-1)
            conditioning = images
        else:
            raise ValueError(f"Unhandled conditioner: {final_cond_type}")

        t0 = time.perf_counter()
        with torch.inference_mode():
            outputs = model.sample(
                conditioning=conditioning,
                batch_size=len(batch_ids),
                generator=torch.Generator(device=device).manual_seed(args.seed + step),
                output_type=args.output_type,
                guidance_scale=args.guidance_scale,
            )
        dt = time.perf_counter() - t0
        print(f"[sample] step={step:05d}  batch={len(batch_ids)}  time={dt:.3f}s")

        flat = flatten_outputs(outputs)
        if len(flat) != len(batch_ids):
            print(f"[WARN] output count {len(flat)} != batch size {len(batch_ids)}; will save min count")
        save_n = min(len(flat), len(batch_targets))

        for k in range(save_n):
            target = batch_targets[k]
            ok = False
            if args.overwrite and target.exists():
                try:
                    target.unlink()
                except Exception:
                    pass
            if args.output_type == "trimesh":
                ok = save_trimesh_item(flat[k], target)
                print(f"[save] {target}" if ok else f"[skip] could not export {target.name}")
            else:
                torch.save(flat[k], target)
                ok = True
                print(f"[save] {target}")

            if manifest_w is not None:
                plan_idx = batch_ids[k]
                source = "csv" if args.csv is not None else "direct"
                csv_row = plan_idx if args.csv is not None else ""
                uid = target.stem

                cond_id = ""
                cond_text = ""
                if final_cond_type == "text":
                    cond_text = (rows[csv_row].get("cap3d_caption") if args.csv is not None
                                 else str(conditioning))
                elif final_cond_type == "lvis":
                    if args.csv is not None:
                        cat = (rows[csv_row].get("lvis_category") or "").strip()
                        cond_text = cat
                        cond_id = str(lvis_map.get(cat, -1)) if (lvis_map is not None) else "-1"
                    else:
                        cond_id = str(conditioning if isinstance(conditioning, int) else batch_cond_vals[k])
                elif final_cond_type == "gobjaverse":
                    if args.csv is not None:
                        try:
                            cond_id = str(int(rows[csv_row].get("class_id", -1)))
                        except Exception:
                            cond_id = "-1"
                    else:
                        cond_id = str(conditioning if isinstance(conditioning, int) else batch_cond_vals[k])
                elif final_cond_type == "image":
                    sprite = (rows[csv_row].get("view_path") if args.csv is not None else batch_cond_vals[k]) or ""
                    cond_text = f"{os.path.basename(sprite)}@{picked_views[k]}"

                manifest_w.writerow({
                    "uid": uid,
                    "out_path": str(target),
                    "ok": int(ok),
                    "cond_type": final_cond_type,
                    "cond_id": cond_id,
                    "cond_text": cond_text,
                    "source": source,
                    "csv_row": csv_row,
                    "seed": args.seed + step,
                    "step": step,
                    "batch_size": len(batch_ids),
                    "output_type": args.output_type,
                    "dt_sec": f"{dt:.3f}",
                })
                manifest_f.flush()

        step += 1

    print(f"[done] wrote {len(pending_indices)} new items; total present: {len(plan)}")

    if manifest_f is not None:
        manifest_f.close()

if __name__ == "__main__":
    main()
