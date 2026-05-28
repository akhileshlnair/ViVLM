#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "mimic_cxr_vqa"
TRAIN_PATH = DATA_DIR / "train" / "mimic_cxr_vqa.jsonl"
MANIFEST_PATH = DATA_DIR / "manifests" / "downloaded_datasets.csv"


@dataclass(frozen=True)
class SplitSpec:
    name: str
    json_file: str
    images_zip: str
    prompt_style: str


TRAIN_SPEC = SplitSpec(
    name="MIMIC-CXR-VQA-train",
    json_file="train.json",
    images_zip="train_images.zip",
    prompt_style="train",
)


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)


def wipe_output() -> None:
    if RAW_DIR.exists():
        shutil.rmtree(RAW_DIR)
    if TRAIN_PATH.exists():
        TRAIN_PATH.unlink()


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        items = [safe_str(v) for v in value if safe_str(v)]
        return ", ".join(items)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).strip()


def append_jsonl(record: dict[str, Any]) -> None:
    with TRAIN_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_manifest(rows_written: int, annotations: int, split: str) -> None:
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
                "dataset": split,
                "source": "MiniMedMind/MIMIC-CXR-VQA",
                "kind": "medical_chest_vqa",
                "rows_written": str(rows_written),
                "train_jsonl": str(TRAIN_PATH.relative_to(ROOT)),
                "notes": f"{annotations} annotations expanded into 4 prompt variants per example",
            }
        )


def prompt_variants(question: str) -> list[str]:
    return [
        f"<image>\nQuestion: {question}\nAnswer briefly with the medically correct response.",
        f"<image>\nPlease answer this chest X-ray question concisely:\n{question}",
        f"<image>\n{question}\nProvide the correct answer in one short phrase.",
        f"<image>\nBased on the chest radiograph, answer the question directly:\n{question}",
    ]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Mirror MIMIC-CXR-VQA into trainable JSONL + image files.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of annotations to mirror.")
    args = parser.parse_args()

    ensure_dirs()
    wipe_output()

    json_path = Path(
        hf_hub_download(
            "MiniMedMind/MIMIC-CXR-VQA",
            TRAIN_SPEC.json_file,
            repo_type="dataset",
        )
    )


def image_uri(member_name: str) -> str:
    return f"data/raw/mimic_cxr_vqa/train_images.zip::{member_name}"
    zip_path = Path(
        hf_hub_download(
            "MiniMedMind/MIMIC-CXR-VQA",
            TRAIN_SPEC.images_zip,
            repo_type="dataset",
        )
    )
    zip_dest = RAW_DIR / "train_images.zip"
    if not zip_dest.exists():
        shutil.copy2(zip_path, zip_dest)

    with json_path.open(encoding="utf-8") as f:
        payload = json.load(f)
    annotations = payload["annotations"]

    with zipfile.ZipFile(zip_path) as archive:
        members = [m for m in archive.infolist() if not m.is_dir() and m.filename.lower().endswith(".jpg")]
        if len(members) != len(annotations):
            raise RuntimeError(
                f"image count mismatch: {len(members)} zip members vs {len(annotations)} annotations"
            )

        total_records = 0
        for idx, (ann, member) in enumerate(zip(annotations, members)):
            if args.limit is not None and idx >= args.limit:
                break

            question = safe_str(ann.get("question"))
            answer = safe_str(ann.get("caption"))
            image_id = safe_str(ann.get("image_id")) or f"train_{idx}.jpg"
            image_ref = image_uri(member.filename)

            for variant_idx, prompt in enumerate(prompt_variants(question)):
                append_jsonl(
                    {
                        "id": f"mimic_cxr_vqa_train_{idx:06d}_{variant_idx:02d}",
                        "source_dataset": "MiniMedMind/MIMIC-CXR-VQA",
                        "split": "train",
                        "kind": "medical_chest_vqa",
                        "image": image_ref,
                        "prompt": prompt,
                        "response": answer,
                        "answer": answer,
                        "metadata": {
                            "annotation_index": str(idx),
                            "image_id": image_id,
                            "answer_type": safe_str(ann.get("answer_type")),
                            "source_image_member": member.filename,
                        },
                    }
                )
                total_records += 1

            if (idx + 1) % 1000 == 0:
                print(f"[MIMIC-CXR-VQA] {idx + 1} annotations -> {total_records} records")

    annotations_written = args.limit if args.limit is not None else len(annotations)
    write_manifest(total_records, annotations_written, TRAIN_SPEC.name)
    print(f"[MIMIC-CXR-VQA] wrote {total_records} records")


if __name__ == "__main__":
    main()
