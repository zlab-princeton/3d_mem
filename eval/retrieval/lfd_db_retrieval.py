import os, hashlib
import argparse
import json
import time
from pathlib import Path
import tempfile
import torch
import numpy as np
from tqdm import tqdm

from lfd_utils.lfd import FastLFDMetric
from lfd_utils.lfd_batch_loader import LFDBatchLoader


def worker(
    proc_id: int,
    gpu_id: int,
    query_indices_chunk: list[int],
    args: argparse.Namespace,
    temp_dir: Path,
):
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)
    name = f"[W{proc_id}|GPU{gpu_id}]"

    print(f"{name} loading LFD train DB to {device} …")
    train_db_dir = Path(args.train_db_dir)
    test_db_dir  = Path(args.test_db_dir)

    metric = FastLFDMetric(
        precomputed_db_paths={
            "art":  train_db_dir / "lfd_db_art.pt",
            "fd":   train_db_dir / "lfd_db_fd.pt",
            "cir":  train_db_dir / "lfd_db_cir.pt",
            "ecc":  train_db_dir / "lfd_db_ecc.pt",
            "meta": train_db_dir / "lfd_db_meta.json",
        },
        device=device,
        tgt_chunk=args.tgt_chunk,
    )

    # Globals (q8/al10) live on device in the metric already, but we sometimes want them here too
    loader = LFDBatchLoader(device=device)
    globals_metric = metric.get_global_tensors()
    globals_loader = loader.get_global_tensors()
    torch.testing.assert_close(globals_metric["q8"],  globals_loader["q8"])
    torch.testing.assert_close(globals_metric["al10"], globals_loader["al10"])

    print(f"{name} loading TEST LFD feature tensors (CPU-mem mapped) …")
    test_feats_cpu = {
        "art": torch.load(test_db_dir / "lfd_db_art.pt", map_location="cpu"),
        "fd":  torch.load(test_db_dir / "lfd_db_fd.pt",  map_location="cpu"),
        "cir": torch.load(test_db_dir / "lfd_db_cir.pt", map_location="cpu"),
        "ecc": torch.load(test_db_dir / "lfd_db_ecc.pt", map_location="cpu"),
    }
    with open(test_db_dir / "lfd_db_meta.json", "r") as f:
        test_meta = json.load(f)

    results = {}
    B = max(1, int(args.batch_size))

    for s in tqdm(range(0, len(query_indices_chunk), B), desc=name, position=proc_id, leave=False):
        batch_idxs = query_indices_chunk[s : s + B]

        for q_idx in batch_idxs:
            q_info = test_meta[q_idx]
            q_id = q_info.get("model_id") or q_info.get("path")

            # Build per-query dict on GPU with correct shapes/dtypes; code tensors as uint8
            q_tensors = {
                "art":  test_feats_cpu["art"][q_idx].unsqueeze(0).to(device=device, dtype=torch.uint8),
                "fd":   test_feats_cpu["fd"][q_idx].unsqueeze(0).to(device=device, dtype=torch.uint8),
                "cir":  test_feats_cpu["cir"][q_idx].unsqueeze(0).to(device=device, dtype=torch.uint8),
                "ecc":  test_feats_cpu["ecc"][q_idx].unsqueeze(0).to(device=device, dtype=torch.uint8),
                "q8":   globals_metric["q8"],    # int32 on device
                "al10": globals_metric["al10"],  # int32 on device
            }

            try:
                topk = metric.distance_from_tensors(q_tensors, top_k=args.topk)
                results[q_id] = topk
            except Exception as e:
                print(f"{name} ERROR on {q_id}: {e}")
                results[q_id] = []

    out_path = temp_dir / f"results_{proc_id}.json"
    with open(out_path, "w") as f:
        json.dump(results, f)
    print(f"{name} saved {len(results)} results to {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Streamed, memory-bounded LFD retrieval (Top-K).")
    ap.add_argument("--train_db_dir", required=True)
    ap.add_argument("--test_db_dir",  required=True)
    ap.add_argument("--output",       required=True)
    ap.add_argument("--topk", type=int, default=1)
    ap.add_argument("--tgt-chunk", type=int, default=256, help="Targets per chunk for streamed topK")
    ap.add_argument("--batch-size", type=int, default=32, help="Query micro-batch per GPU")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("[ERROR] No CUDA device available.")
        return

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    tokens = [t for t in cvd.split(",") if t]

    if tokens:
        if all(tok.strip().isdigit() for tok in tokens):
            gpus = [int(tok.strip()) for tok in tokens]
        else:
            gpus = list(range(len(tokens)))
    else:
        gpus = list(range(torch.cuda.device_count()))

    print(f"[INFO] Using GPUs (local ordinals): {gpus}")
    with open(Path(args.test_db_dir) / "lfd_db_meta.json", "r") as f:
        num_queries = len(json.load(f))
    all_q = list(range(num_queries))
    print(f"[INFO] Total queries: {len(all_q)}")

    buckets = [[] for _ in range(len(gpus))]
    for i, q in enumerate(all_q):
        buckets[i % len(gpus)].append(q)

    import torch.multiprocessing as mp
    ctx = mp.get_context("spawn")

    start = time.time()
    with tempfile.TemporaryDirectory(prefix="lfd_stream_") as tdir:
        tdir_p = Path(tdir)
        procs = []
        for i, gpu in enumerate(gpus):
            if not buckets[i]:
                continue
            p = ctx.Process(target=worker, args=(i, gpu, buckets[i], args, tdir_p))
            p.start()
            procs.append(p)

        for p in procs:
            p.join()
            if p.exitcode != 0:
                print("[ERROR] A worker crashed; aborting.")
                return

        merged = {}
        for i in range(len(gpus)):
            part = tdir_p / f"results_{i}.json"
            if part.exists():
                with open(part, "r") as f:
                    merged.update(json.load(f))

    outp = Path(os.path.expanduser(args.output)).resolve()
    outp.parent.mkdir(parents=True, exist_ok=True)

    tmp = outp.with_suffix(outp.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(merged, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, outp) 

    with open(outp, "r") as f:
        check = json.load(f)
        
    dur = time.time() - start
    print(f"\n[✓] Done in {dur:.2f}s — wrote {len(check)} results → {outp}")
    if len(check):
        k = next(iter(check))
        print(f"[info] sample key: {k} → {check[k][:1]}")

if __name__ == "__main__":
    main()
