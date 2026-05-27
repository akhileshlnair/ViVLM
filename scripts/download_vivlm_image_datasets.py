#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from itertools import islice
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
TRAIN_DIR = DATA_DIR / "train"
MANIFEST_DIR = DATA_DIR / "manifests"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    source: str
    split: str
    image_column: str = "image"
    label_candidates: tuple[str, ...] = ("label", "labels", "dx", "diagnosis", "condition", "caption")
    prompt: str = "Look at the patient's body-part image and provide the most likely diagnosis."
    kind: str = "medical_image"


SPECS: list[DatasetSpec] = [
    DatasetSpec(
        name="PAD-UFES-20",
        source="SalmaneExploring/pad-ufes-20",
        split="train",
        label_candidates=("label", "dx", "diagnosis", "lesion", "class", "lesion_type"),
        prompt="Look at the skin-lesion photo and provide the most likely diagnosis.",
        kind="smartphone_skin",
    ),
    DatasetSpec(
        name="SCIN",
        source="google/scin",
        split="train",
        label_candidates=("labels", "label", "condition", "condition_name", "caption", "description"),
        prompt="Look at the dermatology photo and give the most likely condition.",
        kind="consumer_dermatology",
    ),
    DatasetSpec(
        name="FLUO-SC",
        source="Matheusbecali/FLUO-SC",
        split="train",
        label_candidates=("label", "labels", "class", "dx"),
        prompt="Look at the skin-lesion photo and give the most likely diagnosis.",
        kind="smartphone_skin",
    ),
    DatasetSpec(
        name="SkinCAP",
        source="joshuachou/SkinCAP",
        split="train",
        label_candidates=("caption", "medical_caption", "description", "label", "labels"),
        prompt="Describe the dermatology image and give the likely diagnosis if one is explicit in the source data.",
        kind="captioned_dermatology",
    ),
    DatasetSpec(
        name="oral-images",
        source="GPrabhanjana/oral-images",
        split="train",
        label_candidates=("label", "labels", "class", "diagnosis"),
        prompt="Look at the oral-cavity photo and provide the most likely diagnosis.",
        kind="oral_lesions",
    ),
]


def slug(name: str) -> str:
    return (
        name.lower()
        .replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
        .replace(".", "_")
    )


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, list):
        items = [as_text(item) for item in value if as_text(item)]
        return ", ".join(items)
    return str(value).strip()


def first_nonempty(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        if key in row:
            text = as_text(row.get(key))
            if text:
                return text
    return ""


def resolve_label(ds, row: dict[str, Any], spec: DatasetSpec) -> str:
    for key in spec.label_candidates:
        if key not in row:
            continue
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            text = ", ".join(as_text(item) for item in value if as_text(item))
            if text:
                return text
        if isinstance(value, int):
            feature = ds.features.get(key) if hasattr(ds.features, "get") else None
            names = getattr(feature, "names", None)
            if names and 0 <= value < len(names):
                return names[value]
        text = as_text(value)
        if text:
            return text
    return "unknown"


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)


def write_manifest_row(path: Path, row: dict[str, Any]) -> None:
    header_needed = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if header_needed:
            writer.writeheader()
        writer.writerow(row)


def image_extension(image: Image.Image) -> str:
    fmt = (image.format or "").lower()
    if fmt in {"jpeg", "jpg"}:
        return ".jpg"
    if fmt in {"png"}:
        return ".png"
    if fmt in {"webp"}:
        return ".webp"
    if fmt in {"tif", "tiff"}:
        return ".tiff"
    return ".jpg"


def save_image(image: Image.Image, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    image = image.convert("RGB")
    image.save(dest, quality=95, optimize=True)


def normalize_dataset(spec: DatasetSpec, limit: int | None = None) -> None:
    if limit is None:
        ds = load_dataset(spec.source, split=spec.split)
        rows_iter = (ds[idx] for idx in range(len(ds)))
    else:
        ds = load_dataset(spec.source, split=spec.split, streaming=True)
        rows_iter = islice(ds, limit)
    ds_dir = RAW_DIR / slug(spec.name)
    train_path = TRAIN_DIR / f"{slug(spec.name)}.jsonl"
    manifest_path = MANIFEST_DIR / "downloaded_datasets.csv"

    count = 0
    for idx, row in enumerate(rows_iter):
        count = idx + 1
        image = row.get(spec.image_column)
        if image is None:
            continue
        if not hasattr(image, "save"):
            # Some datasets return dicts with image bytes/paths. Best-effort fallback.
            if isinstance(image, dict) and "path" in image and image["path"]:
                from PIL import Image as PILImage

                image = PILImage.open(image["path"])
            else:
                continue

        label = resolve_label(ds, row, spec)

        image_path = ds_dir / "images" / f"{idx:06d}{image_extension(image)}"
        save_image(image, image_path)

        prompt = spec.prompt
        response = label

        record = {
            "id": f"{slug(spec.name)}_{idx:06d}",
            "source_dataset": spec.source,
            "split": spec.split,
            "image": str(image_path.relative_to(ROOT)),
            "prompt": prompt,
            "response": response,
            "kind": spec.kind,
        }

        extras = {}
        for key, value in row.items():
            if key == spec.image_column:
                continue
            text = as_text(value)
            if text and key not in {"label", "labels", "dx", "diagnosis", "condition", "caption"}:
                extras[key] = text
        if extras:
            record["metadata"] = extras

        with train_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    write_manifest_row(
        manifest_path,
        {
            "dataset": spec.name,
            "source": spec.source,
            "split": spec.split,
            "rows_written": str(count),
            "kind": spec.kind,
            "train_jsonl": str(train_path.relative_to(ROOT)),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and normalize high-signal medical image datasets for ViVLM.")
    parser.add_argument("--only", nargs="*", help="Optional dataset names to download.")
    parser.add_argument("--limit", type=int, default=None, help="Optional row cap per dataset.")
    args = parser.parse_args()

    ensure_dirs()
    selected = SPECS if not args.only else [spec for spec in SPECS if spec.name in set(args.only)]
    if not selected:
        raise SystemExit("No matching datasets selected.")

    for spec in selected:
        print(f"[ViVLM] downloading {spec.name} from {spec.source} ({spec.split})")
        normalize_dataset(spec, limit=args.limit)


if __name__ == "__main__":
    main()
