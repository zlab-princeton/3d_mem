import argparse
from pathlib import Path
from multiprocessing import Pool
from functools import partial
import trimesh
import os
import sys
from tqdm import tqdm
import gc

# Add parent directory to path to find lfd_utils
sys.path.append(str(Path(__file__).parent))

try:
    from lfd_utils.lfd_me import MeshEncoder
except ImportError as e:
    raise ImportError("Required module `lfd_utils.lfd_me` not found. Please check your installation.") from e

LFD_SUFFIXES = ["_q4_v1.8.art", "_q8_v1.8.art", "_q8_v1.8.cir", "_q8_v1.8.ecc", "_q8_v1.8.fd"]


def get_outdir_and_prefix(geometry_file: Path) -> tuple[Path, str]:
    g = Path(geometry_file)
    stem = g.stem.lower()
    generic_exact = {"model", "model_normalized", "mesh"}

    if stem in generic_exact:
        prefix = g.parent.name
        out_dir = g.parent / "lfd_feature"
    else:
        prefix = g.stem
        out_dir = g.parent / "lfd_feature" / prefix

    return out_dir, prefix

def all_lfd_files_exist(lfd_dir: Path, prefix: str) -> bool:
    """Checks if all LFD-related files for a given prefix exist in lfd_dir."""
    return lfd_dir.exists() and all((lfd_dir / f"{prefix}{suffix}").exists() for suffix in LFD_SUFFIXES)


def encode_single_mesh_to_lfd(geometry_file: Path, force_recompute=False):
    """Encodes a mesh to LFD features"""
    geometry_file = Path(geometry_file)
    if not geometry_file.exists():
        return False, str(geometry_file), "File not found"

    lfd_output_dir, model_id = get_outdir_and_prefix(geometry_file)

    if not force_recompute and all_lfd_files_exist(lfd_output_dir, model_id):
        return True, str(geometry_file), "Already exists"

    lfd_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        mesh = trimesh.load(str(geometry_file), process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        if mesh.is_empty or mesh.faces is None or len(mesh.faces) == 0:
            try:
                mesh = mesh.as_trimesh()
            except Exception:
                pass
            if mesh.is_empty or mesh.faces is None or len(mesh.faces) == 0:
                return False, str(geometry_file), "No usable faces after load/triangulate"

        encoder = MeshEncoder(mesh.vertices, mesh.faces, folder=lfd_output_dir, file_name=model_id)
        encoder.align_mesh()

        if all_lfd_files_exist(lfd_output_dir, model_id):
            return True, str(geometry_file), "Success"
        else:
            return False, str(geometry_file), "Missing output"
    except Exception as e:
        return False, str(geometry_file), f"Exception: {e}"




def main():

    parser = argparse.ArgumentParser(description="Multiprocessing LFD extractor from a file list.")
    parser.add_argument("--force_recompute", action="store_true", help="Force recomputation even if files exist.")
    parser.add_argument("--n_process", type=int, default=10, help="Number of processes to use.")
    parser.add_argument("--file_list", type=str, default="NoObjaverse_files.txt", help="Path to the text file containing a list of mesh files.")
    args = parser.parse_args()

    file_list_path = Path(args.file_list)
    print(f"[INFO] Reading file paths from: {file_list_path}")

    if not file_list_path.is_file():
        print(f"[ERROR] File list '{file_list_path}' not found.")
        print("[INFO] Please run the 'create_file_list.py' script first to generate it.")
        return

    with open(file_list_path, 'r') as f:
        geometry_files = [Path(line.strip()) for line in f if line.strip()]

    if not geometry_files:
        print(f"[ERROR] No file paths were found in {file_list_path}.")
        return

    print(f"[INFO] Found {len(geometry_files)} files to process.")
    print(f"[INFO] Starting LFD encoding with {args.n_process} processes...")

    # Start processing
    process_fn = partial(encode_single_mesh_to_lfd, force_recompute=args.force_recompute)
    success, fail = 0, 0

    with Pool(args.n_process) as pool:
        results = pool.imap_unordered(process_fn, geometry_files)
        
        for ok, path, message in tqdm(results, total=len(geometry_files), desc="Processing models", unit="file"):
            if ok:
                success += 1
                # tqdm.write(f"[✓] {path}  →  {message}") 
            else:
                # Optionally print failures as they happen for immediate feedback
                # print(f"[✗] {path}  →  {message}")
                fail += 1
                tqdm.write(f"[✗] {path}  →  {message}")

    print("\n📊 LFD Multiprocess Summary:")
    print(f"   Success : {success}")
    print(f"   Failures: {fail}")
    print(f"   Total   : {success + fail}")

if __name__ == "__main__":
    main()

