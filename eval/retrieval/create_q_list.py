import argparse
from pathlib import Path
from tqdm import tqdm


MESH_EXTS = ['*.obj', '*.glb', '*.gltf']
LFD_SUFFIXES = ["_q8_v1.8.art", "_q8_v1.8.cir", "_q8_v1.8.ecc", "_q8_v1.8.fd"]
UNI3D_FILE = "uni3d_embedding.npy"

def _is_valid_lfd_dir(d: Path) -> bool:
    """Checks if a directory contains a complete set of LFD files."""
    if not d.is_dir(): return False
    # Find any .fd file to get the prefix (e.g. "mesh_q8_v1.8.fd")
    for fd_file in d.glob("*_q8_v1.8.fd"):
        prefix = fd_file.name[:-len("_q8_v1.8.fd")]
        # Check if all corresponding suffixes exist
        if all((d / f"{prefix}{suf}").exists() for suf in LFD_SUFFIXES):
            return True
    return False

def _scan_lfd(root: Path) -> list[Path]:
    """Finds directories containing complete LFD features."""
    leaf_dirs = set()
    for lfd_root in tqdm(list(root.rglob("lfd_feature")), desc="Scanning LFD"):
        for sub in lfd_root.iterdir():
            if _is_valid_lfd_dir(sub):
                leaf_dirs.add(sub.resolve())

        if _is_valid_lfd_dir(lfd_root):
            leaf_dirs.add(lfd_root.resolve())
    return sorted(leaf_dirs)

def main():
    parser = argparse.ArgumentParser(description="Scan directories for Mesh, Uni3D, or LFD paths.")
    parser.add_argument("--input_dir", "-i", required=True, nargs='+', help="Root directories to scan.")
    parser.add_argument("--output_txt", "-o", required=True, help="Output .txt file path.")
    parser.add_argument("--type", "-t", required=True, choices=["mesh", "lfd", "uni3d"])
    args = parser.parse_args()

    paths_to_write = []

    for input_dir in args.input_dir:
        root = Path(input_dir).resolve()
        if not root.exists():
            print(f"[WARN] Skipping missing directory: {root}")
            continue

        print(f"Scanning {root} for '{args.type}'...")

        if args.type == "mesh":
            for ext in MESH_EXTS:
                paths_to_write.extend(root.rglob(ext))
        
        elif args.type == "uni3d":
            paths_to_write.extend(root.rglob(UNI3D_FILE))
        
        elif args.type == "lfd":
            paths_to_write.extend(_scan_lfd(root))

    final_paths = sorted(set(str(p.resolve()) for p in paths_to_write))

    out_path = Path(args.output_txt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, 'w') as f:
        f.write("\n".join(final_paths))
        if final_paths: f.write("\n")

    print(f"Found {len(final_paths)} items. Saved to {out_path}")

if __name__ == "__main__":
    main()