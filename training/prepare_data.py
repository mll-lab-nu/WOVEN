"""Convert WOVEN question parquet files into SFT (qwen-vl-finetune) or GRPO (verl) training data.

Usage:
    # SFT data from the Hugging Face dataset, with 4x option-permutation augmentation
    python training/prepare_data.py --split train --out data/sft --permute 4

    # Controlled subset (any combination of action / reasoning types)
    python training/prepare_data.py --split train --out data/sft_perc_causal --permute 4 \
        --action-types perceptive --reasoning-types forward_dynamics inverse_dynamics

    # GRPO parquet (training) and a 200-item validation parquet
    python training/prepare_data.py --split train --out data/rl --format verl
    python training/prepare_data.py --split val --out data/rl --format verl --limit 200

    # Local parquet files instead of the hub
    python training/prepare_data.py --local-parquet path/to/train/questions-*.parquet --out data/sft

Outputs:
    --format sft   <out>/<split>.json      qwen-vl-finetune conversation list; image paths are relative
                   <out>/images/<id>_<k>.jpg  to the JSON file (k = 1-based index of <image_k>);
                                           JPEG bytes are copied from the parquet unchanged.
    --format verl  <out>/<split>.parquet   verl RLHF parquet with images embedded as bytes.

Prompt format: question images first (one <image> block per occurrence of <image_k> in the question,
each occurrence replaced by "[image above]"), then the question, "\\n\\nOptions:", image options as
"\\n<image>Option X" and text options as "\\nOption X: <text>", and the final line
"Answer with ONLY the option letter (A, B, C, or D)." One image is attached per <image> block, so an
image referenced twice (for example a static option equal to the question image) is attached twice.
"""
import argparse
import glob
import json
import os
import re
import sys

import pyarrow as pa
import pyarrow.parquet as pq

HF_REPO = "MLL-Lab/WOVEN"
LETTERS = "ABCD"
ANSWER_INSTRUCTION = "Answer with ONLY the option letter (A, B, C, or D)."
IMAGE_REF = re.compile(r"<image_(\d+)>")
DATA_SOURCE = "woven"

ACTION_TYPES = ("exogenous", "perceptive", "inspective", "navigative", "manipulative")
REASONING_TYPES = (
    "forward_dynamics", "inverse_dynamics", "counterfactual_removal", "counterfactual_substitution",
    "outcome_prediction", "cued_prediction", "temporal_ordering", "temporal_adjacency",
)
READ_COLUMNS = ["id", "action_type", "reasoning_type", "question", "options", "answer", "images"]


# ---------------------------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------------------------
def render(question: str, options: list[dict]) -> tuple[str, list[int]]:
    """Render one item in the assess prompt format.

    Returns the user text (with one "<image>" token per attached image) and the list of 0-based
    indices into the item's `images` list, in the order the <image> tokens appear.
    """
    image_order: list[int] = []

    def question_ref(m: re.Match) -> str:
        image_order.append(int(m.group(1)) - 1)
        return "[image above]"

    question_text = IMAGE_REF.sub(question_ref, question)
    parts = ["<image>" * len(image_order), question_text.rstrip(), "\n\nOptions:"]
    for opt in options:
        if opt["is_image"]:
            m = IMAGE_REF.fullmatch(opt["content"].strip())
            if m is None:
                raise ValueError(f"image option without <image_k> reference: {opt['content']!r}")
            image_order.append(int(m.group(1)) - 1)
            parts.append(f"\n<image>Option {opt['label']}")
        else:
            parts.append(f"\nOption {opt['label']}: {opt['content']}")
    parts.append(f"\n{ANSWER_INSTRUCTION}")
    text = "".join(parts)
    assert text.count("<image>") == len(image_order)
    return text, image_order


def permute(row: dict, n: int) -> list[dict]:
    """Option-permutation augmentation.

    For each target letter (A, B, C, D in that order, first `n` of them) the correct option is moved
    to that position and the three distractors fill the remaining positions in their original
    relative order. The copy whose target equals the original answer keeps the original id; the
    other copies get the suffix `_perm<letter>`. Applies to image and text options alike.
    """
    if n <= 1:
        return [row]
    options = row["options"]
    gold = row["answer"]
    correct = next(o for o in options if o["label"] == gold)
    distractors = [o for o in options if o["label"] != gold]
    out = []
    for target in LETTERS[:n]:
        rest = iter(distractors)
        new_options = []
        for label in LETTERS[: len(options)]:
            src = correct if label == target else next(rest)
            new_options.append({**src, "label": label, "is_correct": label == target})
        new_id = row["id"] if target == gold else f"{row['id']}_perm{target}"
        out.append({**row, "id": new_id, "options": new_options, "answer": target})
    return out


# ---------------------------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------------------------
def resolve_parquet_files(args) -> list[str]:
    if args.local_parquet:
        files = []
        for pattern in args.local_parquet:
            matched = sorted(glob.glob(pattern))
            if not matched:
                sys.exit(f"no parquet files match {pattern!r}")
            files.extend(matched)
        return files
    from huggingface_hub import snapshot_download

    pattern = f"{args.split}/questions-*.parquet"
    local = snapshot_download(repo_id=args.hf_repo, repo_type="dataset", allow_patterns=[pattern],
                              token=os.environ.get("HF_TOKEN"))
    files = sorted(glob.glob(os.path.join(local, pattern)))
    if not files:
        sys.exit(f"no files matching {pattern} in {args.hf_repo}")
    return files


def iter_rows(files: list[str], batch_size: int = 64):
    for path in files:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=batch_size, columns=READ_COLUMNS):
            yield from batch.to_pylist()


def parse_list(values: list[str] | None, allowed: tuple[str, ...], flag: str) -> set[str] | None:
    if not values:
        return None
    out = {v.strip() for item in values for v in item.split(",") if v.strip()}
    unknown = out - set(allowed)
    if unknown:
        sys.exit(f"{flag}: unknown value(s) {sorted(unknown)}; allowed: {list(allowed)}")
    return out


# ---------------------------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------------------------
VERL_SCHEMA = pa.schema([
    ("data_source", pa.string()),
    ("prompt", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
    ("images", pa.list_(pa.struct([("bytes", pa.binary())]))),
    ("ability", pa.string()),
    ("reward_model", pa.struct([("style", pa.string()), ("ground_truth", pa.string())])),
    ("extra_info", pa.struct([("id", pa.string())])),
])


def image_bytes(img) -> bytes:
    return img["bytes"] if isinstance(img, dict) else img


def write_images(row: dict, image_dir: str) -> list[str]:
    """Write each image of a source item once as <image_dir>/<id>_<k>.jpg; return the paths."""
    paths = []
    for k, img in enumerate(row["images"], start=1):
        path = os.path.join(image_dir, f"{row['id']}_{k}.jpg")
        if not os.path.exists(path):
            tmp = path + ".tmp"
            with open(tmp, "wb") as f:
                f.write(image_bytes(img))
            os.replace(tmp, path)
        paths.append(path)
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train", help="split directory on the hub (train, val, ...)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--format", choices=("sft", "verl"), default="sft")
    ap.add_argument("--permute", type=int, default=1, choices=(1, 2, 3, 4),
                    help="number of option permutations per item (4 = correct option at each of A-D)")
    ap.add_argument("--action-types", nargs="+", help=f"keep only these action types {ACTION_TYPES}")
    ap.add_argument("--reasoning-types", nargs="+", help=f"keep only these reasoning types {REASONING_TYPES}")
    ap.add_argument("--limit", type=int, default=0, help="keep only the first N items after filtering")
    ap.add_argument("--local-parquet", nargs="+", help="local question parquet files or globs (skips the hub)")
    ap.add_argument("--hf-repo", default=HF_REPO)
    ap.add_argument("--name", default=None, help="output file stem (default: the split name)")
    args = ap.parse_args()

    action_types = parse_list(args.action_types, ACTION_TYPES, "--action-types")
    reasoning_types = parse_list(args.reasoning_types, REASONING_TYPES, "--reasoning-types")
    files = resolve_parquet_files(args)
    stem = args.name or args.split
    os.makedirs(args.out, exist_ok=True)

    if args.format == "sft":
        out_path = os.path.join(args.out, f"{stem}.json")
        image_dir = os.path.join(args.out, "images")
        os.makedirs(image_dir, exist_ok=True)
        json_dir = os.path.dirname(os.path.abspath(out_path))
        records = []
    else:
        out_path = os.path.join(args.out, f"{stem}.parquet")
        writer = pq.ParquetWriter(out_path + ".tmp", VERL_SCHEMA)
        buffer = []

    n_src = n_out = 0
    for row in iter_rows(files):
        if action_types and row["action_type"] not in action_types:
            continue
        if reasoning_types and row["reasoning_type"] not in reasoning_types:
            continue
        if args.limit and n_src >= args.limit:
            break
        n_src += 1
        if args.format == "sft":
            src_paths = [os.path.relpath(p, json_dir) for p in write_images(row, image_dir)]
        for item in permute(row, args.permute):
            text, order = render(item["question"], item["options"])
            if args.format == "sft":
                records.append({
                    "id": item["id"],
                    "image": [src_paths[i] for i in order],
                    "conversations": [
                        {"from": "human", "value": text},
                        {"from": "gpt", "value": item["answer"]},
                    ],
                })
            else:
                buffer.append({
                    "data_source": DATA_SOURCE,
                    "prompt": [{"role": "user", "content": text}],
                    "images": [{"bytes": image_bytes(row["images"][i])} for i in order],
                    "ability": "world_model",
                    "reward_model": {"style": "rule", "ground_truth": item["answer"]},
                    "extra_info": {"id": item["id"]},
                })
                if len(buffer) >= 256:
                    writer.write_table(pa.Table.from_pylist(buffer, schema=VERL_SCHEMA))
                    buffer = []
            n_out += 1

    if args.format == "sft":
        with open(out_path + ".tmp", "w") as f:
            json.dump(records, f)
        os.replace(out_path + ".tmp", out_path)
    else:
        if buffer:
            writer.write_table(pa.Table.from_pylist(buffer, schema=VERL_SCHEMA))
        writer.close()
        os.replace(out_path + ".tmp", out_path)
    print(f"source items {n_src} -> training items {n_out} -> {out_path}")


if __name__ == "__main__":
    main()
