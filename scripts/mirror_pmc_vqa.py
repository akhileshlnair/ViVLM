#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import shutil
from io import BytesIO
from pathlib import Path
from typing import Any

from datasets import load_dataset
from huggingface_hub import hf_hub_url, list_repo_files
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "pmc_vqa"
TRAIN_PATH = DATA_DIR / "train" / "pmc_vqa.jsonl"
MANIFEST_PATH = DATA_DIR / "manifests" / "downloaded_datasets.csv"


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)


def wipe_output() -> None:
    if RAW_DIR.exists():
        shutil.rmtree(RAW_DIR)
    if TRAIN_PATH.exists():
        TRAIN_PATH.unlink()


def append_jsonl(record: dict[str, Any]) -> None:
    with TRAIN_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        parts = [safe_str(v) for v in value if safe_str(v)]
        return ", ".join(parts)
    return str(value).strip()


def save_image(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, quality=95, optimize=True)


def write_manifest(rows_written: int, shards: int) -> None:
    header_needed = not MANIFEST_PATH.exists()
    with MANIFEST_PATH.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "source", "kind", "rows_written", "train_jsonl", "notes"],
        )
        if header_needed:
            writer.writeheader()
        writer.writerow(
            {
                "dataset": "PMC-VQA",
                "source": "OctoMed/PMC-VQA",
                "kind": "medical_vqa",
                "rows_written": str(rows_written),
                "train_jsonl": str(TRAIN_PATH.relative_to(ROOT)),
                "notes": f"expanded from {shards} parquet shards with 16 reasoning traces per case",
            }
        )


def shard_files() -> list[str]:
    files = list_repo_files("OctoMed/PMC-VQA", repo_type="dataset")
    return sorted(f for f in files if f.startswith("data/train-") and f.endswith(".parquet"))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Mirror PMC-VQA into trainable JSONL + image files.")
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--start-shard", type=int, default=1, help="1-based shard index to start from.")
    parser.add_argument("--append", action="store_true", help="Append to existing outputs instead of wiping them.")
    args = parser.parse_args()

    ensure_dirs()
    if not args.append:
        wipe_output()

    total_records = 0
    shard_list = shard_files()
    shard_list = shard_list[args.start_shard - 1 :]
    if args.max_shards is not None:
        shard_list = shard_list[: args.max_shards]

    for shard_idx, shard in enumerate(shard_list, start=args.start_shard):
        url = hf_hub_url("OctoMed/PMC-VQA", shard, repo_type="dataset")
        ds = load_dataset("parquet", data_files=url, split="train", streaming=True)
        case_count = 0
        for row_idx, row in enumerate(ds):
            if args.max_cases is not None and total_records // 16 >= args.max_cases:
                break
            image = row["image"]
            if image is None:
                continue
            case_id = safe_str(row.get("id")) or f"{Path(shard).stem}_{row_idx:06d}"
            img_path = RAW_DIR / Path(shard).stem / f"{case_id}.jpg"
            save_image(image, img_path)

            question = safe_str(row.get("question"))
            options = row.get("options") or []
            answer = safe_str(row.get("answer"))
            responses = row.get("responses") or []
            prompt_lines = [question]
            if options:
                prompt_lines.append("Options:")
                prompt_lines.extend([f"- {safe_str(option)}" for option in options])
            prompt = "\n".join(prompt_lines)

            if not responses:
                responses = [answer]

            for resp_idx, response in enumerate(responses):
                append_jsonl(
                    {
                        "id": f"pmc_vqa_{shard_idx:02d}_{row_idx:06d}_{resp_idx:02d}",
                        "source_dataset": "OctoMed/PMC-VQA",
                        "kind": "medical_vqa",
                        "image": str(img_path.relative_to(ROOT)),
                        "prompt": prompt,
                        "response": safe_str(response) or answer,
                        "answer": answer,
                        "metadata": {
                            "case_id": case_id,
                            "image_hash": safe_str(row.get("image_hash")),
                            "shard": shard,
                            "response_index": str(resp_idx),
                        },
                    }
                )
                total_records += 1
            case_count += 1
        print(f"[PMC-VQA] shard {shard_idx}/{len(shard_list)} -> {case_count} cases, {total_records} records total")

    write_manifest(total_records, len(shard_list))
    print(f"[PMC-VQA] wrote {total_records} records")


if __name__ == "__main__":
    main()
