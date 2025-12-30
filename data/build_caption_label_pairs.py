import argparse, csv, json, math, random, sys, subprocess
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from collections import Counter

def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def canon(s: str) -> str:
    return " ".join(s.lower().split())

def read_lines(p: Path) -> List[str]:
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]

def wc_count_lines(p: Path) -> Optional[int]:
    try:
        out = subprocess.check_output(["wc", "-l", str(p)], text=True)
        return int(out.strip().split()[0])
    except Exception:
        try:
            with p.open("rb") as f:
                return sum(1 for _ in f)
        except Exception:
            return None

CLASSIFY_TASK = "Given an object description, classify it into one of the fine-grained object classes."
def make_query_text(desc: str) -> str:
    return f"Instruct: {CLASSIFY_TASK}\nThe description is: {desc}"

LABEL_TEMPLATES = [
    "class label: {name}",
    "label: {name}",
    "{name}",
]
def make_label_prompts(name: str) -> List[str]:
    return [tpl.format(name=name) for tpl in LABEL_TEMPLATES]

def build_model(model_name: str, device: str, dtype: str, use_flash: bool):
    torch_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[dtype]
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    kw = {}
    if use_flash:
        kw["attn_implementation"] = "flash_attention_2"
    try:
        model = AutoModel.from_pretrained(model_name, dtype=torch_dtype, **kw).to(device)
    except TypeError:
        model = AutoModel.from_pretrained(model_name, torch_dtype=torch_dtype, **kw).to(device)
    model.eval()
    return tokenizer, model

@torch.inference_mode()
def encode_texts(tokenizer, model, texts: List[str], max_length: int, batch_size: int, device: str,
                 progress_desc: Optional[str] = None) -> Tensor:
    if not isinstance(texts, list):
        texts = list(texts)
    n = len(texts)
    indices = range(0, n, batch_size)
    if progress_desc:
        indices = tqdm(indices, desc=progress_desc, dynamic_ncols=True)
    vecs: List[Tensor] = []
    cur_bs = batch_size
    i_iter = iter(indices)
    while True:
        try:
            i = next(i_iter)
        except StopIteration:
            break
        chunk = texts[i:i+cur_bs]
        try:
            batch = tokenizer(chunk, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            emb = last_token_pool(out.last_hidden_state, batch["attention_mask"])
            emb = F.normalize(emb, p=2, dim=1)
            vecs.append(emb)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) and cur_bs > 8:
                torch.cuda.empty_cache()
                cur_bs = max(8, cur_bs // 2)
                remaining = list(range(i, n, cur_bs))
                if progress_desc:
                    i_iter = iter(tqdm(remaining, desc=f"{progress_desc} (mb={cur_bs})", dynamic_ncols=True))
                else:
                    i_iter = iter(remaining)
                continue
            raise
    return torch.cat(vecs, dim=0) if vecs else torch.empty(0, model.config.hidden_size, device=device)

@torch.inference_mode()
def build_label_bank(tokenizer, model, labels: List[str], max_length: int, batch_size: int, device: str,
                     show_progress: bool=False) -> Tuple[Tensor, List[str], List[str]]:
    prompts, spans = [], []
    for name in labels:
        ps = make_label_prompts(name)
        spans.append((len(prompts), len(prompts) + len(ps)))
        prompts.extend(ps)
    E = encode_texts(tokenizer, model, prompts, max_length=max_length, batch_size=batch_size, device=device,
                     progress_desc=("encode labels" if show_progress else None))  # [sumP, D]
    rows = []
    for s, e in spans:
        rows.append(F.normalize(E[s:e].mean(dim=0, keepdim=True), p=2, dim=1))
    bank = torch.cat(rows, dim=0)  # [C, D]
    labels_can = [canon(x) for x in labels]
    return bank, labels, labels_can

def load_bank_npz(npz_path: Path, device: str) -> Tuple[Tensor, List[str], List[str], dict]:
    data = np.load(npz_path, allow_pickle=True)
    bank = torch.tensor(data["bank"]).to(device)
    meta = json.loads(str(data["metadata"]))
    return bank, list(meta["labels_raw"]), list(meta["labels_canonical"]), meta


class Stats:
    def __init__(self, nbins=50):
        self.nbins=nbins
        self.total_rows=0; self.empty_caps=0
        self.evaluated_rows=0; self.kept_rows=0
        self.per_class=Counter()
        self.hist_all=[0]*nbins; self.hist_keep=[0]*nbins
    def _bin(self,s): s=max(0.0,min(0.999999,float(s))); return int(s*self.nbins)
    def add(self,sims,idxs,keep,labels):
        for s,j,k in zip(sims,idxs,keep):
            self.evaluated_rows+=1
            b=self._bin(s); self.hist_all[b]+=1
            if k:
                self.kept_rows+=1; self.hist_keep[b]+=1; self.per_class[labels[j]]+=1

def write_reports(stats: Stats, labels_used: list, out_prefix: Path, meta: dict):
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    with (out_prefix.with_suffix(".counts.tsv")).open("w", encoding="utf-8") as f:
        for k,v in sorted(stats.per_class.items(), key=lambda kv:-kv[1]):
            f.write(f"{k}\t{v}\n")
    with (out_prefix.with_suffix(".hist.tsv")).open("w", encoding="utf-8") as f:
        f.write("bin_left\tbin_right\tall\tkept\n"); nb=stats.nbins
        for i in range(nb):
            bl=i/nb; br=(i+1)/nb
            f.write(f"{bl:.3f}\t{br:.3f}\t{stats.hist_all[i]}\t{stats.hist_keep[i]}\n")
    summary = {
        "total_rows": stats.total_rows,
        "empty_captions": stats.empty_caps,
        "evaluated_rows": stats.evaluated_rows,
        "kept_rows": stats.kept_rows,
        "dropped_rows": max(0, stats.evaluated_rows - stats.kept_rows),
        "acceptance_rate": (stats.kept_rows / stats.evaluated_rows) if stats.evaluated_rows else 0.0,
        "labels_used": len(labels_used),
        **meta,
    }
    with (out_prefix.with_suffix(".summary.json")).open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[ok] wrote {out_prefix.with_suffix('.counts.tsv')}")
    print(f"[ok] wrote {out_prefix.with_suffix('.hist.tsv')}")
    print(f"[ok] wrote {out_prefix.with_suffix('.summary.json')}", flush=True)


def main():
    ap = argparse.ArgumentParser("Classify Cap3D captions with Qwen3-Embedding-8B (Transformers, streaming, stats)")

    ap.add_argument("--labels", help="Path to lvis_top100.txt or lvis_min50.txt (ignored if --bank_npz provided)")
    ap.add_argument("--bank_npz", help="Optional precomputed label bank .npz (with metadata)")
    
    # data
    ap.add_argument("--csv", required=True, help="Cap3D Objaverse caption CSV (uid,caption; no header)")
    ap.add_argument("--out_csv", required=True, help="Output CSV (uid,caption,label,sim)")
    
    ap.add_argument("--model_name", default="Qwen/Qwen3-Embedding-8B")
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--dtype", choices=["fp32","bf16","fp16"], default="fp32")
    ap.add_argument("--use_flash", action="store_true", help="Enable flash_attention_2 if available")
    ap.add_argument("--max_length", type=int, default=512, help="Token limit; captions are short so 512 is enough")
    ap.add_argument("--batch", type=int, default=512, help="Embedding micro-batch (auto-reduces on OOM)")
    
    ap.add_argument("--threshold", type=float, default=0.70)
    ap.add_argument("--log_every", type=int, default=40960)
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--stats_prefix", default=None, help="Prefix for stats tsv/json; defaults to out_csv without extension")
    args = ap.parse_args()

    if not args.bank_npz and not args.labels:
        print("[error] Provide --labels or --bank_npz", file=sys.stderr); sys.exit(1)

    tokenizer, model = build_model(args.model_name, args.device, args.dtype, args.use_flash)
    print(f"[devices] model={args.model_name} device={args.device} dtype={args.dtype}")
    
    meta = {}
    if args.bank_npz:
        bank, labels_raw, labels_can, meta_npz = load_bank_npz(Path(args.bank_npz), args.device)
        meta.update({"bank_source": "npz", **{f"npz_{k}": v for k,v in meta_npz.items() if k != "labels_raw" and k != "labels_canonical"}})
        print(f"[bank] loaded from {args.bank_npz}  rows={bank.shape[0]} dim={bank.shape[1]}")
    else:
        labels = read_lines(Path(args.labels))
        print(f"[bank] building from labels: {args.labels} (n={len(labels)})")
        bank, labels_raw, labels_can = build_label_bank(tokenizer, model, labels, max_length=args.max_length, batch_size=args.batch,
                                                        device=args.device, show_progress=args.progress)
        meta.update({"bank_source": "labels_txt", "labels_txt": args.labels})
    assert bank.shape[0] == len(labels_raw)

    csv_in = Path(args.csv)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    total_lines = wc_count_lines(csv_in) if args.progress else None

    stats = Stats(nbins=50)
    kept_so_far = 0
    processed = 0
    start_t = torch.cuda.Event(enable_timing=True) if args.device.startswith("cuda") else None
    end_t = torch.cuda.Event(enable_timing=True) if args.device.startswith("cuda") else None
    wall_start = torch.cuda.Event(enable_timing=False) if args.device.startswith("cuda") else None

    if args.progress:
        pbar = tqdm(total=total_lines, unit="rows", dynamic_ncols=True)

    with csv_in.open("r", newline="") as fin, out_csv.open("w", newline="", encoding="utf-8") as fout:
        r = csv.reader(fin)
        w = csv.writer(fout)
        w.writerow(["uid", "caption", "label", "sim"])

        buf_uids: List[str] = []
        buf_caps: List[str] = []

        def flush_batch():
            nonlocal kept_so_far, processed
            if not buf_caps:
                return
            q_texts = [make_query_text(c) for c in buf_caps]
            Q = encode_texts(tokenizer, model, q_texts, max_length=args.max_length, batch_size=args.batch,
                             device=args.device, progress_desc=None)
            # cosine sim vs bank
            S = Q @ bank.T
            vals, idxs = torch.max(S, dim=1)
            sims = vals.detach().cpu().tolist()
            idxs = idxs.detach().cpu().tolist()
            keep_mask = [s >= args.threshold for s in sims]
            stats.add(sims, idxs, keep_mask, labels_raw)

            kept_batch = 0
            for uid, cap, j, s, keep in zip(buf_uids, buf_caps, idxs, sims, keep_mask):
                if keep:
                    w.writerow([uid, cap, labels_raw[j], f"{s:.6f}"])
                    kept_batch += 1

            kept_so_far += kept_batch
            processed += len(buf_caps)

        for row in r:
            if not row:
                if args.progress: pbar.update(1)
                continue
            stats.total_rows += 1
            uid = row[0]
            cap = row[1] if len(row) > 1 else ""
            if not cap.strip():
                stats.empty_caps += 1
                if args.progress: pbar.update(1)
                continue
            buf_uids.append(uid); buf_caps.append(cap)
            if len(buf_caps) >= args.batch:
                flush_batch()
                if args.progress:
                    pbar.update(len(buf_caps))
                if processed % max(1, args.log_every) == 0:
                    dt = max(1e-6, (processed / (max(1, processed))))
                    acc = kept_so_far / max(1, stats.evaluated_rows)
                    print(f"[progress] proc={processed:,} kept={kept_so_far:,} acc_rate={acc:.4f}", flush=True)
                buf_uids.clear(); buf_caps.clear()

        flush_batch()
        if args.progress:
            pbar.update(len(buf_caps))
            pbar.close()

    print(f"[classify] total={stats.total_rows} empty={stats.empty_caps} "
          f"evaluated={stats.evaluated_rows} kept={stats.kept_rows} "
          f"accept_rate={stats.kept_rows/max(1,stats.evaluated_rows):.4f} (thr={args.threshold:.3f})", flush=True)

    prefix = Path(args.stats_prefix) if args.stats_prefix else out_csv.with_suffix("")
    meta.update({
        "threshold": args.threshold,
        "bank_rows": int(bank.shape[0]),
        "label_templates": LABEL_TEMPLATES,
        "model_name": args.model_name,
        "dtype": args.dtype,
        "max_length": args.max_length,
        "batch": args.batch,
        "csv_in": str(csv_in),
    })
    write_reports(stats, labels_raw, prefix, meta)

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
