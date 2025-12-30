from __future__ import annotations

import argparse, json, os, sys, tempfile, subprocess
from pathlib import Path
from typing import List, Dict, Tuple, DefaultDict
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm

from lfd_utils.lfd_batch_loader import LFDBatchLoader

ART_SUFFIX = "_q8_v1.8.art"

def db_exists(out_dir: Path) -> bool:
    return (out_dir / "lfd_db_meta.json").exists()

def load_lines(p: Path) -> List[Path]:
    with p.open("r", encoding="utf-8") as f:
        return [Path(x.strip()) for x in f if x.strip()]

def has_art(dir_path: Path) -> bool:
    try:
        return any(dir_path.glob(f"*{ART_SUFFIX}"))
    except Exception:
        return False

def safe_ancestor(p: Path, up: int) -> Path:
    q = p
    for _ in range(up):
        if q.parent == q:
            break
        q = q.parent
    return q

def derive_meta(dir_path: Path, *, relbase: Path | None, infer_shapenet: bool, model_id_level: int) -> Dict:
    model_node = safe_ancestor(dir_path, model_id_level)
    model_id = model_node.name
    category_id = None
    if infer_shapenet:
        cat_node = safe_ancestor(model_node, 1)
        category_id = None if cat_node == model_node else cat_node.name
    if relbase is not None:
        try:
            relpath = str(dir_path.relative_to(relbase))
        except ValueError:
            relpath = str(dir_path)
    else:
        relpath = str(dir_path)
    return {"model_id": model_id, "category_id": category_id, "relpath": relpath}


def _child_load_one(dir_path: Path, device: str, out_npz: Path):
    """Run in a separate interpreter to avoid propagating segfaults."""
    loader = LFDBatchLoader(device=device)
    # warm up globals
    _ = loader.get_global_tensors()
    t = loader.get(dir_path)
    if t is None:
        raise RuntimeError("LFDBatchLoader.get returned None")
    # Save numpy arrays (int32) so parent can read without torch
    np.savez_compressed(
        out_npz,
        art=t["art"].detach().to("cpu").numpy().astype(np.int32, copy=False),
        fd =t["fd" ].detach().to("cpu").numpy().astype(np.int32, copy=False),
        cir=t["cir"].detach().to("cpu").numpy().astype(np.int32, copy=False),
        ecc=t["ecc"].detach().to("cpu").numpy().astype(np.int32, copy=False),
    )

def child_entry() -> bool:
    if os.environ.get("LFD_CHILD") != "1":
        return False
    dir_path = Path(os.environ["LFD_DIR"])
    device   = os.environ.get("LFD_DEVICE","cpu")
    out_npz  = Path(os.environ["LFD_NPZ"])
    try:
        _child_load_one(dir_path, device, out_npz)
    except Exception as e:
        print(f"[CHILD-ERROR] {dir_path}: {e}", file=sys.stderr)
        sys.exit(2)
    sys.exit(0)


def build_db_for_dirs(
    dirs: List[Path],
    *,
    out_dir: Path,
    relbase: Path | None,
    infer_shapenet: bool,
    device: str,
    isolate: bool,
    workers: int,
    model_id_level: int,
) -> Tuple[int, Tuple[int, ...], Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
    out_dir.mkdir(parents=True, exist_ok=True)

    dirs = [d for d in dirs if d.is_dir() and has_art(d)]
    if not dirs:
        print("[WARN] No valid feature dirs for this group; skipping.")
        return 0, (), (), (), ()

    print(f"[INFO] Building LFD DB at {out_dir} from {len(dirs)} directories (isolate={isolate}, workers={workers})")

    try:
        LFDBatchLoader(device=device).get_global_tensors()
    except Exception as e:
        print(f"[FATAL] get_global_tensors failed in parent: {e}", file=sys.stderr)
        return 0, (), (), (), ()

    art_list, fd_list, cir_list, ecc_list = [], [], [], []
    meta_list: List[Dict] = []
    ok = 0

    if not isolate:
        # Fast path
        loader = LFDBatchLoader(device=device)
        _ = loader.get_global_tensors()
        it = tqdm(dirs, desc="Loading LFD dirs")
        for d in it:
            try:
                t = loader.get(d)
                if t is None:
                    continue
                art_list.append(t["art"].detach().to("cpu"))
                fd_list.append( t["fd" ].detach().to("cpu"))
                cir_list.append(t["cir"].detach().to("cpu"))
                ecc_list.append(t["ecc"].detach().to("cpu"))
                meta_list.append(derive_meta(d, relbase=relbase, infer_shapenet=infer_shapenet, model_id_level=model_id_level))
                ok += 1
            except Exception as e:
                print(f"[WARN] load failed for {d}: {e}", file=sys.stderr)
    else:
        pend: List[Tuple[subprocess.Popen, Path, Path]] = []

        def reap_finished(block: bool = False):
            nonlocal ok
            while True:
                progressed = False
                i = 0
                while i < len(pend):
                    p, d, npz = pend[i]
                    rc = p.poll()
                    if rc is None:
                        i += 1
                        continue
                    stdout, stderr = p.communicate()
                    pend.pop(i)
                    if rc == 0 and npz.exists():
                        try:
                            z = np.load(npz)
                            art_list.append(torch.from_numpy(z["art"]))
                            fd_list.append( torch.from_numpy(z["fd"]))
                            cir_list.append(torch.from_numpy(z["cir"]))
                            ecc_list.append(torch.from_numpy(z["ecc"]))
                            meta_list.append(derive_meta(d, relbase=relbase, infer_shapenet=infer_shapenet, model_id_level=model_id_level))
                            ok += 1
                        except Exception as e:
                            print(f"[WARN] failed to read child npz for {d}: {e}", file=sys.stderr)
                    try:
                        npz.unlink(missing_ok=True)
                    except Exception:
                        pass
                    progressed = True
                if progressed:
                    return
                if not block:
                    return
                if pend:
                    pend[0][0].wait()
                else:
                    return

        import uuid
        tmp_root = Path(tempfile.mkdtemp(prefix="lfd_batch_"))
        for d in tqdm(dirs, desc="Loading LFD dirs (isolated)"):
            npz = tmp_root / f"{uuid.uuid4().hex}.npz"
            env = os.environ.copy()
            env.update({
                "LFD_CHILD": "1",
                "LFD_DIR": str(d),
                "LFD_DEVICE": device,
                "LFD_NPZ": str(npz),
                "PYTHONPATH": os.pathsep.join([str(Path(".")), env.get("PYTHONPATH","")]),
            })
            p = subprocess.Popen([sys.executable, __file__], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            pend.append((p, d, npz))
            while len(pend) >= workers:
                reap_finished(block=True)
        while pend:
            reap_finished(block=True)

    if ok == 0:
        print("[WARN] No directories loaded successfully for this group; skipping.")
        return 0, (), (), (), ()

    art = torch.cat(art_list, dim=0)
    fd  = torch.cat(fd_list,  dim=0)
    cir = torch.cat(cir_list, dim=0)
    ecc = torch.cat(ecc_list, dim=0)

    torch.save(art, out_dir / "lfd_db_art.pt")
    torch.save(fd,  out_dir / "lfd_db_fd.pt")
    torch.save(cir, out_dir / "lfd_db_cir.pt")
    torch.save(ecc, out_dir / "lfd_db_ecc.pt")
    with (out_dir / "lfd_db_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta_list, f)

    print(f"[✓] Saved DB at {out_dir}: N={ok}")
    print(f"    - {out_dir/'lfd_db_art.pt'}  {tuple(art.shape)}")
    print(f"    - {out_dir/'lfd_db_fd.pt'}   {tuple(fd.shape)}")
    print(f"    - {out_dir/'lfd_db_cir.pt'}  {tuple(cir.shape)}")
    print(f"    - {out_dir/'lfd_db_ecc.pt'}  {tuple(ecc.shape)}")
    print(f"    - {out_dir/'lfd_db_meta.json'}")

    return ok, tuple(art.shape), tuple(fd.shape), tuple(cir.shape), tuple(ecc.shape)


def discover_feature_dirs(db_root: Path) -> List[Path]:
    seen = set()
    leaves: List[Path] = []
    for art_file in db_root.glob(f"**/*{ART_SUFFIX}"):
        parent = art_file.parent
        if parent.is_dir() and parent not in seen:
            seen.add(parent)
            leaves.append(parent)
    return leaves


def main():
    if child_entry():
        return

    ap = argparse.ArgumentParser("Build LFD DB(s) from a list of feature dirs and/or a db_root tree (robust)")

    # Sources (you may provide either or both). If both are provided,
    # script will use the list as the source of leaf dirs and group relative to db_root.
    ap.add_argument("--gen_list", type=Path, help="TXT file with absolute paths to leaf feature dirs (one per line)")
    ap.add_argument("--db_root",  type=Path, help="Root directory to scan for leaf feature dirs (used for discovery and/or grouping)")
    
    ap.add_argument("--out_dir",  type=Path, required=True, help="Output root directory for DB(s)")
    ap.add_argument("--relbase",  type=Path, default=None, help="Base path to make meta.relpath relative to (default: db_root if provided)")

    # Grouping controls (db_root mode)
    ap.add_argument("--keep_level", type=int, default=1, help="How many leading path components (relative to --db_root) to keep per-DB grouping")

    # Metadata controls
    ap.add_argument("--model_id_level", type=int, default=0, help="0: use leaf feature dir name as model_id; 1: parent; 2: grandparent; ...")
    ap.add_argument("--infer_shapenet", action="store_true", help="Derive category_id as one level above the chosen model_id node")

    # Runtime controls
    ap.add_argument("--device", type=str, default="cuda", choices=["cpu","cuda"])
    ap.add_argument("--isolate", action="store_true", help="Isolate each load in a subprocess (recommended).")
    ap.add_argument("--no-isolate", dest="isolate", action="store_false")
    ap.add_argument("--workers", type=int, default=1, help="Parallel children when --isolate (N processes)")
    ap.set_defaults(isolate=True)

    ap.add_argument("--overwrite", action="store_true",
                    help="Rebuild DB(s) even if the output directory already has a DB (detected via lfd_db_meta.json).")

    args = ap.parse_args()

    out_root: Path = args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)

    if args.gen_list is None and args.db_root is None:
        print("[ERROR] Provide --gen_list and/or --db_root")
        sys.exit(1)
        
    relbase: Path | None = args.relbase or args.db_root

    total_ok = 0
    group_count = 0
    skipped_groups = 0

    leaf_dirs: List[Path] = []
    if args.gen_list is not None:
        lst = load_lines(args.gen_list)
        if not lst:
            print("[ERROR] Empty --gen_list"); sys.exit(1)
        leaf_dirs.extend(lst)
        print(f"[INFO] Loaded {len(lst)} leaf dirs from --gen_list")

    if args.db_root is not None and args.gen_list is None:
        db_root: Path = args.db_root
        if not db_root.exists() or not db_root.is_dir():
            print(f"[ERROR] --db_root not found or not a dir: {db_root}")
            sys.exit(1)
        discovered = discover_feature_dirs(db_root)
        print(f"[INFO] Discovered {len(discovered)} leaf dirs under --db_root")
        leaf_dirs.extend(discovered)

    if not leaf_dirs:
        print("[ERROR] No candidate leaf dirs to process")
        sys.exit(1)


    seen = set()
    uniq_leaf_dirs: List[Path] = []
    for d in leaf_dirs:
        if d not in seen:
            seen.add(d)
            uniq_leaf_dirs.append(d)
    leaf_dirs = uniq_leaf_dirs

    do_group = args.db_root is not None and (args.keep_level is not None and int(args.keep_level) > 0)

    if not do_group:
        if not args.overwrite and db_exists(out_root):
            print(f"[SKIP] DB already exists at {out_root} (use --overwrite to rebuild)")
            skipped_groups += 1
        else:
            ok, *_ = build_db_for_dirs(
                leaf_dirs,
                out_dir=out_root,
                relbase=relbase,
                infer_shapenet=args.infer_shapenet,
                device=args.device,
                isolate=args.isolate,
                workers=args.workers,
                model_id_level=args.model_id_level,
            )
            if ok > 0:
                total_ok += ok
                group_count = 1
    else:
        db_root = args.db_root
        K = max(0, int(args.keep_level))
        groups: DefaultDict[Tuple[str, ...], List[Path]] = defaultdict(list)
        skipped = 0
        for d in leaf_dirs:
            try:
                rel = d.relative_to(db_root)
                parts = rel.parts
                key = tuple(parts[:K]) if K <= len(parts) else tuple(parts)
                groups[key].append(d)
            except ValueError:
                skipped += 1
        if skipped:
            print(f"[WARN] Skipped {skipped} dirs not under --db_root when grouping")

        print(f"[INFO] Grouping {sum(len(v) for v in groups.values())} dirs into {len(groups)} group(s) with keep_level={K}")

        for key, dirs in sorted(groups.items()):
            out_dir = out_root.joinpath(*key) if key else out_root
            if not args.overwrite and db_exists(out_dir):
                print(f"[SKIP] {out_dir} already has a DB (use --overwrite to rebuild)")
                skipped_groups += 1
                continue
            ok, *_ = build_db_for_dirs(
                dirs,
                out_dir=out_dir,
                relbase=relbase,
                infer_shapenet=args.infer_shapenet,
                device=args.device,
                isolate=args.isolate,
                workers=args.workers,
                model_id_level=args.model_id_level,
            )
            if ok > 0:
                total_ok += ok
                group_count += 1

    if total_ok == 0:
        if skipped_groups > 0:
            print(f"[DONE] No DBs built because {skipped_groups} target dir(s) already contain DB(s). Use --overwrite to rebuild.")
            sys.exit(0)
        print("[ERROR] No directories loaded successfully.")
        sys.exit(3)

    print(f"[DONE] Built {group_count} DB(s); total rows loaded: {total_ok}")

if __name__ == "__main__":
    main()
