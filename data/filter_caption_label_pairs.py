import argparse, csv, json, re, sys, time
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

SYSTEM_PROMPT = """You are a strict data screener for single-object 3D captions.

TASK:
Given (label, caption), decide if the caption predominantly and concretely describes ONE instance of the object class == label.

STRICT RUBRIC (all must be true for KEEP=YES):
1) SINGLE OBJECT: Caption is about a single object instance, not a set/row/collection/scene/room/environment.
2) LABEL DOMINATES: The main described entity is the label; the label isn't just a small part of something else.
3) NOT PART-ONLY: The description is not mainly a part-of-object (e.g., “table leg”, “door handle”), unless the label itself is such a part class.
4) WHEN UNSURE, SAY NO. It is better to kill the innocent by mistake than to let the guilty go free.

SIMPLIFY CAPTION (if KEEP=YES):
- Remove color and material words.
- Keep geometry/parts/structure words that help identify the object shape.

OUTPUT FORMAT (MUST be valid one-line JSON):
{"keep": "yes"|"no", "simple_caption": "<string or empty if no>"}
"""

USER_TEMPLATE = """Label: {label}
Caption: {caption}
Return ONLY the JSON line as specified.
"""

ATTENTION_SUFFIX = """\n\nATTENTION: Return exactly ONE line of VALID JSON with keys "keep" and "simple_caption".
No prose. No markdown. No extra text. If unsure, set "keep":"no" and "simple_caption":"".
"""

COLOR_WORDS = r"(?i)\b(white|black|red|green|blue|yellow|purple|violet|orange|pink|brown|grey|gray|silver|gold|beige|maroon|navy|teal|cyan|magenta|turquoise)\b"
MATERIAL_WORDS = r"(?i)\b(wood|wooden|metal|metallic|steel|iron|aluminum|aluminium|plastic|rubber|leather|glass|ceramic|stone|concrete|fabric|cloth|carbon( fiber)?|bronze|brass|copper|gold(en)?|silver(y)?)\b"

def post_simplify(text: str) -> str:
    text = re.sub(COLOR_WORDS, "", text)
    text = re.sub(MATERIAL_WORDS, "", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text.rstrip(" .,")

def extract_json(s: str) -> Optional[dict]:
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def validate_obj(obj: dict) -> Tuple[bool, str]:
    if not isinstance(obj, dict):
        return False, "not_a_dict"
    if "keep" not in obj or "simple_caption" not in obj:
        return False, "missing_keys"
    keep = str(obj["keep"]).strip().lower()
    if keep not in ("yes", "no"):
        return False, "keep_not_yes_no"
    sc = obj.get("simple_caption", "")
    if sc is None:
        sc = ""
    if not isinstance(sc, str):
        return False, "simple_caption_not_str"
    if keep == "yes":
        if len(sc.strip()) == 0:
            return False, "empty_simple_caption_on_yes"
        if not sc.isascii():
            return False, "non_ascii_caption"
        if re.search(COLOR_WORDS, sc) or re.search(MATERIAL_WORDS, sc):
            return False, "color_or_material_leaked"
    return True, ""

def chunked(iterable, n):
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf

def read_seen_uids(path: Path) -> set:
    seen = set()
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            if r.fieldnames and "uid" in r.fieldnames:
                for row in r:
                    uid = (row.get("uid") or "").strip()
                    if uid:
                        seen.add(uid)
    return seen

def _detect_existing_fieldnames(path: Path) -> List[str]:
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            if r.fieldnames:
                return list(r.fieldnames)
    return ["uid","keep","caption","label"]

def ensure_header(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["uid","keep","caption","label"])
            w.writeheader()

def append_rows(path: Path, rows: List[dict]):
    fieldnames = _detect_existing_fieldnames(path)
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        for r in rows:
            w.writerow(r)
            
def _final_merge(out_csv: str, in_csv: str, final_csv: str):
    out_path = Path(out_csv)
    in_path = Path(in_csv)
    final_path = Path(final_csv)

    uid2label: Dict[str, str] = {}
    with in_path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            u = (row.get("uid") or "").strip()
            lab = (row.get("label") or "").strip()
            if u:
                uid2label[u] = lab

    rows_by_uid: Dict[str, dict] = {}
    with out_path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            u = (row.get("uid") or "").strip()
            if not u:
                continue
            keep = (row.get("keep") or "").strip()
            cap  = (row.get("caption") or "").strip()
            lab  = (row.get("label") or "").strip() if "label" in (r.fieldnames or []) else ""
            if not lab:
                lab = uid2label.get(u, "")
            rows_by_uid[u] = {"uid": u, "keep": keep, "caption": cap, "label": lab}

    with final_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["uid","keep","caption","label"])
        w.writeheader()
        for u, row in rows_by_uid.items():
            w.writerow(row)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--model_name", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--retry_backoff", type=float, default=1.5)
    ap.add_argument("--log_every", type=int, default=1000)
    ap.add_argument("--final_merge_out", default=None,
                  help="Optional: write a deduped 4-col CSV at end. If existing out_csv lacks label, "
                       "labels are joined from in_csv.")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_name)    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    sys_msgs = [{"role": "system", "content": SYSTEM_PROMPT}]

    inp = Path(args.in_csv)
    outp = Path(args.out_csv)
    outp.parent.mkdir(parents=True, exist_ok=True)

    seen_uids = read_seen_uids(outp)
    ensure_header(outp)
    pbar = tqdm(total=None, desc="Screening", dynamic_ncols=True)
    with inp.open("r", newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin) 

        buf_to_write: List[dict] = []
        processed = 0
        total = 0

        current_batch_msgs: List[List[dict]] = []
        current_batch_originals: List[dict] = []

        def flush_buffer():
            nonlocal buf_to_write
            if buf_to_write:
                append_rows(outp, buf_to_write)
                buf_to_write = []

        for row in reader:
            total += 1
            uid = (row.get("uid") or "").strip()
            if not uid or uid in seen_uids:
                continue

            label = (row.get("label") or "").strip()
            caption = (row.get("caption") or "").strip()
            user_msg = USER_TEMPLATE.format(label=label, caption=caption)
            current_batch_msgs.append([*sys_msgs, {"role":"user","content":user_msg}])
            current_batch_originals.append({"uid":uid, "label":label, "caption":caption})

            if len(current_batch_msgs) >= max(1, args.batch):
                results = run_batch(tok, model, current_batch_msgs,
                                    retries=args.retries,
                                    backoff=args.retry_backoff,
                                    max_new=args.max_new_tokens)

                for orig, obj in zip(current_batch_originals, results):
                    uid = orig["uid"]
                    label = orig["label"]
                    if obj is None:
                        keep = "no"; sc = ""
                    else:
                        keep = "yes" if str(obj["keep"]).lower() == "yes" else "no"
                        sc = (obj.get("simple_caption") or "").strip()
                        if keep == "yes":
                            sc = post_simplify(sc)
                            if not sc:
                                sc = orig["label"]
                    if keep == "yes":
                        buf_to_write.append({"uid": uid, "keep": "yes", "caption": sc, "label": label})
                        seen_uids.add(uid)

                flush_buffer()
                processed += len(current_batch_msgs)
                pbar.update(len(current_batch_msgs))
                current_batch_msgs.clear()
                current_batch_originals.clear()

                if args.log_every and processed % args.log_every == 0:
                    print(f"[progress] processed={processed} (skipped={len(seen_uids)} total_seen) total_rows={total}", file=sys.stderr)

        if current_batch_msgs:
            results = run_batch(tok, model, current_batch_msgs,
                                retries=args.retries,
                                backoff=args.retry_backoff,
                                max_new=args.max_new_tokens)
            for orig, obj in zip(current_batch_originals, results):
                uid = orig["uid"]
                label = orig["label"]
                if obj is None:
                    keep = "no"; sc = ""
                else:
                    keep = "yes" if str(obj["keep"]).lower() == "yes" else "no"
                    sc = (obj.get("simple_caption") or "").strip()
                    if keep == "yes":
                        sc = post_simplify(sc)
                        if not sc:
                            sc = orig["label"]
                if keep == "yes":
                    buf_to_write.append({"uid": uid, "keep": "yes", "caption": sc, "label": label})
                    seen_uids.add(uid)
            flush_buffer()
            pbar.update(len(current_batch_msgs))

    
    pbar.close()
    if args.final_merge_out:
        _final_merge(out_csv=args.out_csv, in_csv=args.in_csv, final_csv=args.final_merge_out)
        print(f"[final] wrote merged CSV -> {args.final_merge_out}")
    print("[done]")

def run_batch(tok, model, messages_list: List[List[dict]], retries: int, backoff: float, max_new: int):
    ATT = ATTENTION_SUFFIX
    remaining = list(range(len(messages_list)))
    results: Dict[int, Optional[dict]] = {}
    for attempt in range(retries):
        texts = []
        for i in remaining:
            m = messages_list[i]
            if attempt > 0:
                m = m[:-1] + [{"role": "user", "content": m[-1]["content"] + ATT}]
            texts.append(tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True))

        inputs = tok(texts, return_tensors="pt", padding=True).to(model.device)
        with torch.inference_mode():
            gen = model.generate(
                **inputs,
                max_new_tokens=max_new,
            )

        raw_strs = []
        for b in range(gen.shape[0]):
            out_ids = gen[b, inputs.input_ids.shape[1]:]
            raw_strs.append(tok.decode(out_ids, skip_special_tokens=True))

        next_remaining = []
        for loc, raw in enumerate(raw_strs):
            idx = remaining[loc]
            obj = extract_json(raw)
            ok = False
            if obj is not None:
                ok, _ = validate_obj(obj)
            if ok:
                results[idx] = obj
            else:
                next_remaining.append(idx)
        remaining = next_remaining
        if not remaining:
            break
        time.sleep((backoff ** attempt) * 0.5)

    for idx in remaining:
        results[idx] = None
    return [results[i] for i in range(len(messages_list))]

if __name__ == "__main__":
    main()
