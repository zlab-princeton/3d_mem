import argparse
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from tqdm import tqdm


def sanitize_group_name(parts: tuple[str, ...]) -> str:
    """Make a filesystem-friendly name from a tuple of path parts."""
    if not parts:
        return "root"
    raw = "__".join(parts)
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in raw)


def output_paths_for_group(output_dir: Path, output_prefix: str, group_key: tuple[str, ...]):
    """Return (group_name, matrix_path, meta_path) for a group."""
    group_name = sanitize_group_name(group_key)
    matrix_path = output_dir / f"{output_prefix}__{group_name}.npy"
    meta_path = output_dir / f"{output_prefix}__{group_name}_meta.json"
    return group_name, matrix_path, meta_path


def outputs_exist(matrix_path: Path, meta_path: Path) -> bool:
    """Both consolidated outputs exist?"""
    return matrix_path.exists() and meta_path.exists()


def load_feature(path: Path) -> np.ndarray | None:
    """Load a single-vector embedding; return (D,) float32 or None to skip."""
    try:
        feat = np.load(path)
        if feat.ndim == 1:
            pass
        elif feat.ndim == 2 and feat.shape[0] == 1:
            feat = feat[0]
        else:
            print(f"[WARN] {path} has unexpected shape {feat.shape}; expected (D,) or (1,D); skipping.")
            return None
        return feat.astype(np.float32, copy=False)
    except Exception as e:
        print(f"[WARN] Skipping file {path} due to loading error: {e}")
        return None


def process_group(
    group_key: tuple[str, ...],
    file_paths: list[Path],
    db_root: Path,
    model_id_level: int,
    output_dir: Path,
    output_prefix: str,
    overwrite: bool = False,
    model_id_from: str = "path",
) -> bool:
    """
    Build and save a DB for a single group. Returns True if something was saved.

    model_id_from:
      - "path": use the path component at model_id_level (old behavior).
      - "filename": use the file stem as model_id (for flattened dirs).
    """
    group_label = "/".join(group_key) if group_key else "root"
    group_name, matrix_path, meta_path = output_paths_for_group(output_dir, output_prefix, group_key)
    if not overwrite and outputs_exist(matrix_path, meta_path):
        print(f"[SKIP] Group {group_label}: outputs already exist -> {matrix_path.name}, {meta_path.name}")
        return False

    all_features: list[np.ndarray] = []
    metadata: list[dict] = []
    expected_shape = None

    for p in tqdm(sorted(file_paths), desc=f"Group {group_label}", leave=False):
        feat = load_feature(p)
        if feat is None:
            continue

        if expected_shape is None:
            expected_shape = feat.shape
        elif feat.shape != expected_shape:
            print(f"[WARN] {p} shape {feat.shape} != expected {expected_shape}; skipping.")
            continue

        rel_path = p.relative_to(db_root)

        # Determine model_id
        if model_id_from == "filename":
            model_id = p.stem
        else:  # "path" (default)
            rel_parts = rel_path.parts
            if len(rel_parts) <= model_id_level:
                print(f"[WARN] Skipping {p}, path too short for model_id_level {model_id_level}")
                continue
            model_id = rel_parts[model_id_level]

        all_features.append(feat)
        metadata.append({
            "rel_path": str(rel_path),
            "model_id": model_id,
            "group": group_label,
        })

    if not all_features:
        print(f"[INFO] Group {group_label}: no valid features; skipping save.")
        return False

    final_matrix = np.stack(all_features)

    print(f"[INFO] Saving {final_matrix.shape} to {matrix_path}")
    np.save(matrix_path, final_matrix)

    print(f"[INFO] Saving {len(metadata)} metadata entries to {meta_path}")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return True


def main():
    ap = argparse.ArgumentParser(
        description="Consolidate single-vector .npy embeddings into one or many databases."
    )
    ap.add_argument("--db_root", type=Path, required=True, help="Root directory to search for embedding files.")
    ap.add_argument("--output_dir", type=Path, required=True, help="Directory to save consolidated files.")

    ap.add_argument(
        "--embedding_name",
        type=str,
        default=None,
        help=(
            "Exact filename of the embedding to search for (e.g., 'point_embedding.npy'). "
            "Ignored if --pattern is provided."
        ),
    )
    ap.add_argument(
        "--pattern",
        type=str,
        default=None,
        help=(
            "Glob pattern (relative to --db_root) for embedding files, e.g. '*.npy'. "
            "Use this for flattened dirs like <db_root>/<model_uid>.npy."
        ),
    )

    ap.add_argument(
        "--output_prefix",
        type=str,
        required=True,
        help="Prefix for the output files (e.g., 'pointnet_db').",
    )

    ap.add_argument(
        "--model_id_level",
        type=int,
        default=1,
        help=(
            "Index of model_id in the relative path (used when --model_id_from path). "
            "Use 0 for {root}/{model_id}/... ; "
            "Use 1 for {root}/{category_id}/{model_id}/... (default)"
        ),
    )

    ap.add_argument(
        "--model_id_from",
        type=str,
        choices=["path", "filename"],
        default="path",
        help=(
            "How to derive model_id for metadata. "
            "'path' (default): use path component at --model_id_level. "
            "'filename': use the file stem (for <db_root>/<model_uid>.npy)."
        ),
    )

    ap.add_argument(
        "--keep_level",
        type=int,
        default=0,
        help=(
            "How many leading directory levels under --db_root to keep as group keys. "
            "0 = one DB for all; 1 = per first-level subdir; 2 = per second-level, etc."
        ),
    )
    
    ap.add_argument(
        "--name_substring",
        type=str,
        default=None,
        help="If set, apply a filename substring filter (before grouping).",
    )
    ap.add_argument(
        "--name_mode",
        type=str,
        choices=["include", "exclude"],
        default=None,
        help=(
            "How to use --name_substring: "
            "'include' = keep only files whose *filename* contains it; "
            "'exclude' = keep only files whose filename does NOT contain it."
        ),
    )

    
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate and overwrite existing consolidated files instead of skipping.",
    )

    args = ap.parse_args()

    db_root = args.db_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.pattern:
        search_glob = args.pattern
        print(f"[INFO] Using glob pattern '{search_glob}' to discover embedding files.")
    else:
        if not args.embedding_name:
            ap.error("Either --embedding_name or --pattern must be provided.")
        search_glob = args.embedding_name
        print(f"[INFO] Using exact filename '{search_glob}' to discover embedding files.")

    print(f"[INFO] Discovering '{search_glob}' under {db_root} ...")
    found = sorted(db_root.rglob(search_glob))
    print(f"[INFO] Found {len(found)} items total.")

    if not found:
        print("[ERROR] No embedding files found. Exiting.")
        return

    # Build groups by the first `keep_level` directory parts (relative to db_root)
    groups: dict[tuple[str, ...], list[Path]] = defaultdict(list)
    for p in found:
        if args.name_substring and args.name_mode:
            contains = args.name_substring in p.name
            if args.name_mode == "include" and not contains:
                continue
            if args.name_mode == "exclude" and contains:
                continue

        rel = p.relative_to(db_root)
        parts = rel.parts[:-1]
        if len(parts) < args.keep_level:
            print(f"[WARN] Skipping {p} (depth {len(parts)} < keep_level {args.keep_level})")
            continue
        key = tuple(parts[:args.keep_level])
        groups[key].append(p)
    if not groups:
        print(f"[ERROR] No groups formed for keep_level={args.keep_level}. Nothing to do.")
        return

    print(f"[INFO] Formed {len(groups)} group(s).")
    saved_any = False
    for key, paths in groups.items():
        ok = process_group(
            group_key=key,
            file_paths=paths,
            db_root=db_root,
            model_id_level=args.model_id_level,
            output_dir=output_dir,
            output_prefix=args.output_prefix,
            overwrite=args.overwrite,
            model_id_from=args.model_id_from,
        )
        saved_any = saved_any or ok

    if saved_any:
        print(f"\n[✓] Complete! Databases saved in {output_dir}")
    else:
        print("\n[INFO] Nothing saved (no valid features across groups).")


if __name__ == "__main__":
    main()
