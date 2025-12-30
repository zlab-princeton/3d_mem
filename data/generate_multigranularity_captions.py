import argparse
import csv
import os
import random
from typing import List, Tuple, Iterator, Set, Optional

from PIL import Image
import torch
from transformers import AutoProcessor

try:
    from transformers import Qwen3VLMoeForConditionalGeneration
except Exception:
    Qwen3VLMoeForConditionalGeneration = None

try:
    from transformers import Qwen3VLForConditionalGeneration
except Exception:
    Qwen3VLForConditionalGeneration = None


def parse_args():
    p = argparse.ArgumentParser(description="Batch caption multi-view images with Qwen3-VL (supports MoE A3B).")

    p.add_argument("--metadata_csv", type=str, required=True,
                   help="CSV with at least columns: model_uid, view_path (absolute)")
    p.add_argument("--output_csv", type=str, required=True,
                   help="Output CSV path (model_uid, output)")

    p.add_argument("--uid-col", type=str, default="model_uid",
                   help="Column name for model uid")
    p.add_argument("--path-col", type=str, default="view_path",
                   help="Column name for stacked image absolute path")

    # model configs
    p.add_argument("--model_id", type=str, default="Qwen/Qwen3-VL-8B-Instruct",
                   help="HF model id (default: Qwen/Qwen3-VL-8B-Instruct)")
    p.add_argument("--dtype", type=str, default="auto",
                   choices=["auto", "bfloat16", "float16", "float32"],
                   help="torch_dtype for model weights")
    p.add_argument("--device_map", type=str, default="auto",
                   help="device_map for from_pretrained (e.g., 'auto')")
    p.add_argument("--attn_impl", type=str, default=None,
                   choices=[None, "flash_attention_2", "sdpa"],
                   help="Attention implementation")
    p.add_argument("--offline", action="store_true", default=False,
                   help="Use local cache only (HF offline mode)")
    p.add_argument("--force-gpu0", action="store_true", default=False,
                   help="Force map everything to cuda:0 (overrides device_map)")
    p.add_argument("--cache-dir", type=str, default=None,
                   help="Optional transformers cache dir")

    # views / selection
    p.add_argument("--num-views", type=int, default=12,
                   help="Total views stacked vertically in each input image (V)")
    p.add_argument("--views-per-sample", type=int, default=4,
                   help="Number of views to feed per caption (K >= 4)")
    p.add_argument("--split", type=str, default="val", choices=["train", "val"],
                   help="Selection policy: train=random, val=policy")
    p.add_argument("--val-view-policy", type=str, default="even",
                   help="Policy for selecting views at validation time: "
                        "first | center | even | indices:i,j,k | index:i")

    # generation configs
    p.add_argument("--max-new-tokens", type=int, default=256,
                   help="Max new tokens to generate")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature; 0 = greedy")
    p.add_argument("--num-beams", type=int, default=1,
                   help="Beam search beams; 1 disables beam search")

    p.add_argument("--batch-size", type=int, default=32,
                   help="Batch size for chat-template generation")
    p.add_argument("--skip_bad", action="store_true",
                   help="Skip rows that error instead of raising")
    p.add_argument("--progress", action="store_true",
                   help="Show tqdm progress bar")

    # resume
    p.add_argument("--overwrite", action="store_true", default=False,
                   help="Overwrite output CSV instead of resuming")
    p.add_argument("--fsync-every", type=int, default=1,
                   help="os.fsync the CSV file every N batches (0=never)")
    p.add_argument("--fail-log", type=str, default=None,
                   help="Optional path to append (uid\\terror) for rows that fail with --skip_bad")

    p.add_argument("--seed", type=int, default=42, help="Random seed")

    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)


def pick_model_class(model_id: str):
    """
    Heuristic: MoE/A3B models use Qwen3VLMoeForConditionalGeneration.
    Others use Qwen3VLForConditionalGeneration.
    """
    is_moe = ("A3B" in model_id) or ("MoE" in model_id) or ("-A3B-" in model_id)
    if is_moe and Qwen3VLMoeForConditionalGeneration is not None:
        return Qwen3VLMoeForConditionalGeneration, "moe"
    if Qwen3VLForConditionalGeneration is not None:
        return Qwen3VLForConditionalGeneration, "base"
    raise RuntimeError("Neither Qwen3VLMoeForConditionalGeneration nor Qwen3VLForConditionalGeneration is available.")


def load_model_and_processor(model_id: str,
                             dtype: str = "auto",
                             device_map: str = "auto",
                             attn_impl: Optional[str] = None,
                             offline: bool = False,
                             force_gpu0: bool = False,
                             cache_dir: Optional[str] = None):
    dtype_map = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[dtype]
    dev_map = {"": 0} if force_gpu0 else device_map

    kw = {
        "torch_dtype": torch_dtype,
        "device_map": dev_map,
        "local_files_only": offline,
        "cache_dir": cache_dir,
    }
    if attn_impl is not None:
        kw["attn_implementation"] = attn_impl

    ModelCls, flavor = pick_model_class(model_id)
    print(f"[INFO] Loading ({flavor}) model {model_id} with {kw}")
    model = ModelCls.from_pretrained(model_id, **kw)
    processor = AutoProcessor.from_pretrained(model_id, local_files_only=offline, cache_dir=cache_dir)

    if getattr(model.generation_config, "pad_token_id", None) is None:
        model.generation_config.pad_token_id = processor.tokenizer.eos_token_id

    # IMPORTANT for batched generation
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "left"

    try:
        print("[INFO] hf_device_map:", getattr(model, "hf_device_map", None))
    except Exception:
        pass

    return model, processor


def crop_views(img: Image.Image, V: int) -> List[Image.Image]:
    W, H = img.size
    if H % V != 0:
        raise ValueError(f"Image height {H} not divisible by num_views {V}")
    tile_h = H // V
    views: List[Image.Image] = []
    for i in range(V):
        top = i * tile_h
        bottom = (i + 1) * tile_h
        views.append(img.crop((0, top, W, bottom)))
    return views


def parse_indices(spec: str, V: int) -> List[int]:
    idxs = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            idx = int(tok)
        except Exception:
            continue
        idxs.append(max(0, min(V - 1, idx)))
    return sorted(set(idxs))


def pick_views(V: int, K: int, split: str, policy: str) -> List[int]:
    if K < 4:
        K = 4
    if K > V:
        K = V

    if split == "train":
        return sorted(random.sample(range(V), K))

    pol = (policy or "even").lower()
    if pol == "first":
        return list(range(min(K, V)))
    if pol == "center":
        center = V // 2
        order = [center]
        for d in range(1, V):
            if center - d >= 0:
                order.append(center - d)
            if center + d < V:
                order.append(center + d)
        return sorted(order[:K])
    if pol.startswith("indices:"):
        idxs = parse_indices(pol.split(":", 1)[1], V)
        if len(idxs) >= K:
            return idxs[:K]
        remaining = K - len(idxs)
        even = pick_views(V, remaining, "val", "even")
        return sorted(set(idxs + even))[:K]
    if pol.startswith("index:"):
        idx = parse_indices(pol.split(":", 1)[1], V)
        base = idx[:1] if idx else [0]
        even = pick_views(V, K - 1, "val", "even")
        return sorted(set(base + even))[:K]
    if K == 1:
        return [0]
    xs = [round(i * (V - 1) / (K - 1)) for i in range(K)]
    return sorted(sorted(set(xs))[:K])


def build_prompt() -> str:
    return (
        "You are given multiple views of the SAME object captured from different angles. "
        "Combine evidence across views and describe the object at three granularities."
        "Return ONLY a compact JSON object with these exact keys: "
        "phrase, sentence, paragraph. No extra text. No markdown. No newlines."
        "Constraints: phrase is a concise 3-10 word noun phrase (no punctuation at end). "
        "sentence is 1-2 sentences summarizing main parts, colors, and overall shape. "
        "paragraph is 3-6 sentences covering fine details like materials, textures, "
        "distinctive features, geometry, and any context seen in the background. "
        "Avoid speculation; if something is unknown, say 'unknown'."
        "Example output format: {\"phrase\":\"chair\",\"sentence\":\"A four-legged chair...\",\"paragraph\":\"The seat is flat...\"}"
    )


def build_message(images: List[Image.Image], prompt: str):
    return {
        "role": "user",
        "content": [
            *[{"type": "image", "image": im} for im in images],
            {"type": "text", "text": prompt},
        ],
    }


def read_metadata_rows(metadata_csv: str, uid_col: str, path_col: str) -> Iterator[Tuple[str, str]]:
    with open(metadata_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if uid_col not in reader.fieldnames or path_col not in reader.fieldnames:
            raise ValueError(
                f"CSV must contain columns '{uid_col}' and '{path_col}'. Found: {reader.fieldnames}"
            )
        for row in reader:
            uid = (row.get(uid_col) or "").strip()
            ap = (row.get(path_col) or "").strip()
            if not uid or not ap:
                continue
            if not os.path.isabs(ap):
                raise ValueError(f"view_path must be absolute, got: {ap}")
            yield uid, ap


def generate_batch_captions(
    model,
    processor,
    batch_messages: List[List[dict]],
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    num_beams: int = 1,
) -> List[str]:
    # Ensure left padding for batched generation
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "left"

    inputs = processor.apply_chat_template(
        batch_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    )
    inputs = inputs.to(model.device)

    gen_kwargs = {"max_new_tokens": max_new_tokens}
    if num_beams and num_beams > 1:
        gen_kwargs.update({"num_beams": num_beams, "do_sample": False})
    else:
        if temperature and temperature > 0:
            gen_kwargs.update({"do_sample": True, "temperature": float(temperature)})
        else:
            gen_kwargs.update({"do_sample": False})

    with torch.no_grad():
        generated_ids = model.generate(**inputs, **gen_kwargs)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    texts = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return [" ".join(t.split()) for t in texts]

def load_processed_uids(out_csv: str) -> Set[str]:
    done: Set[str] = set()
    if not os.path.exists(out_csv) or os.path.getsize(out_csv) == 0:
        return done
    with open(out_csv, "r", encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        try:
            header = next(r, None)
        except Exception:
            header = None
        for row in r:
            if not row:
                continue
            uid = (row[0] or "").strip()
            if uid:
                done.add(uid)
    return done


def open_output_writer(out_csv: str, overwrite: bool):
    existed = os.path.exists(out_csv)
    mode = "w" if overwrite or (not existed) else "a"
    csvfile = open(out_csv, mode, newline="", encoding="utf-8")
    writer = csv.writer(csvfile)
    need_header = overwrite or (not existed) or (os.path.getsize(out_csv) == 0)
    if need_header:
        writer.writerow(["model_uid", "output"])
        csvfile.flush()
        try:
            os.fsync(csvfile.fileno())
        except OSError:
            pass
    return csvfile, writer


def append_fail(fail_log: Optional[str], uid: str, err: Exception):
    if not fail_log:
        return
    try:
        with open(fail_log, "a", encoding="utf-8") as fl:
            fl.write(f"{uid}\t{type(err).__name__}: {err}\n")
    except Exception:
        pass


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.progress:
        try:
            from tqdm import tqdm
        except ImportError:
            print("[WARN] tqdm not found; continuing without progress bar. Use pip install tqdm")
            args.progress = False

    model, processor = load_model_and_processor(
        model_id=args.model_id,
        dtype=args.dtype,
        device_map=args.device_map,
        attn_impl=args.attn_impl,
        offline=args.offline,
        force_gpu0=args.force_gpu0,
        cache_dir=args.cache_dir,
    )

    prompt = build_prompt()

    out_dir = os.path.dirname(args.output_csv)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    processed: Set[str] = set()
    if not args.overwrite:
        processed = load_processed_uids(args.output_csv)
        if processed:
            print(f"[RESUME] Found {len(processed)} completed rows in {args.output_csv}")

    written = 0
    skipped = 0
    since_last_fsync = 0

    pending_msgs: List[List[dict]] = []
    pending_uids: List[str] = []

    csvfile, writer = open_output_writer(args.output_csv, args.overwrite)

    def flush_batch():
        nonlocal written, pending_msgs, pending_uids, since_last_fsync
        if not pending_msgs:
            return
        outputs = generate_batch_captions(
            model,
            processor,
            pending_msgs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            num_beams=args.num_beams,
        )
        assert len(outputs) == len(pending_uids)
        for uid, out_text in zip(pending_uids, outputs):
            writer.writerow([uid, out_text])
            written += 1

        pending_msgs.clear()
        pending_uids.clear()

        csvfile.flush()
        since_last_fsync += 1
        if args.fsync_every > 0 and since_last_fsync >= args.fsync_every:
            try:
                os.fsync(csvfile.fileno())
            except OSError:
                pass
            since_last_fsync = 0
        print(f"[BATCH] wrote {written} rows total")

    if args.progress:
        from tqdm import tqdm
        row_iter = tqdm(read_metadata_rows(args.metadata_csv, args.uid_col, args.path_col), desc="Captioning")
    else:
        row_iter = read_metadata_rows(args.metadata_csv, args.uid_col, args.path_col)

    try:
        for uid, img_path in row_iter:
            if uid in processed:
                skipped += 1
                continue
            try:
                if not os.path.exists(img_path):
                    raise FileNotFoundError(f"Image not found: {img_path}")

                with Image.open(img_path) as raw_im:
                    img = raw_im.convert("RGB")

                views = crop_views(img, args.num_views)
                idxs = pick_views(
                    V=len(views),
                    K=args.views_per_sample,
                    split=args.split,
                    policy=args.val_view_policy,
                )
                selected = [views[i] for i in idxs]
                if len(selected) < 4:
                    raise ValueError(f"Selected {len(selected)} views (<4).")

                msg = build_message(selected, prompt)
                pending_msgs.append([msg])
                pending_uids.append(uid)

                if len(pending_msgs) >= args.batch_size:
                    flush_batch()

            except Exception as e:
                append_fail(args.fail_log, uid, e)
                if args.skip_bad:
                    print(f"[ERROR] {uid}: {e}")
                    continue
                else:
                    raise

        flush_batch()
    finally:
        try:
            csvfile.flush()
            try:
                os.fsync(csvfile.fileno())
            except OSError:
                pass
            csvfile.close()
        except Exception:
            pass

    print(f"[DONE] Wrote {written} new rows to {args.output_csv} (skipped {skipped} already-done)")


if __name__ == "__main__":
    main()
