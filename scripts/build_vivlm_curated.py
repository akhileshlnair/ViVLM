#!/usr/bin/env python3
from __future__ import annotations

import ast
import csv
import json
import math
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from huggingface_hub import hf_hub_download, list_repo_files
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
TRAIN_DIR = DATA_DIR / "train"
MANIFEST_DIR = DATA_DIR / "manifests"
CACHE_DIR = DATA_DIR / "cache"


@dataclass(frozen=True)
class MirrorSpec:
    name: str
    source: str
    kind: str
    prompt: str


SPECS = {
    "oral-images": MirrorSpec(
        name="oral-images",
        source="GPrabhanjana/oral-images",
        kind="oral_lesions",
        prompt="Look at the oral-cavity photo and provide the most likely diagnosis.",
    ),
    "PAD-UFES-20": MirrorSpec(
        name="PAD-UFES-20",
        source="SalmaneExploring/pad-ufes-20",
        kind="smartphone_skin",
        prompt="Look at the skin-lesion photo and provide the most likely diagnosis.",
    ),
    "FLUO-SC": MirrorSpec(
        name="FLUO-SC",
        source="Matheusbecali/FLUO-SC",
        kind="smartphone_skin",
        prompt="Look at the smartphone skin-lesion photo and provide the most likely diagnosis.",
    ),
    "SCIN": MirrorSpec(
        name="SCIN",
        source="google/scin",
        kind="consumer_dermatology",
        prompt="Look at the dermatology photo and provide the most likely diagnosis.",
    ),
}


def slug(name: str) -> str:
    return (
        name.lower()
        .replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
        .replace(".", "_")
    )


def ensure_dirs() -> None:
    for path in [RAW_DIR, TRAIN_DIR, MANIFEST_DIR, CACHE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def wipe_output(dataset_slug: str) -> tuple[Path, Path]:
    raw_path = RAW_DIR / dataset_slug
    train_path = TRAIN_DIR / f"{dataset_slug}.jsonl"
    if raw_path.exists():
        shutil.rmtree(raw_path)
    if train_path.exists():
        train_path.unlink()
    return raw_path, train_path


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_manifest(rows: list[dict[str, Any]]) -> None:
    manifest_path = MANIFEST_DIR / "downloaded_datasets.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "source", "kind", "rows_written", "train_jsonl", "notes"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_image_bytes(image_bytes: bytes, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image.save(dest, quality=95, optimize=True)


def save_image_file(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(src).convert("RGB")
    image.save(dest, quality=95, optimize=True)


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    if isinstance(value, (list, tuple)):
        parts = [safe_str(v) for v in value if safe_str(v)]
        return ", ".join(parts)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).strip()


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = safe_str(value).lower()
    return text in {"true", "1", "yes", "y"}


def first_nonempty(*values: Any) -> str:
    for value in values:
        text = safe_str(value)
        if text:
            return text
    return ""


def parse_weighted_label(raw: Any) -> str:
    text = safe_str(raw)
    if not text:
        return ""
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, dict) and parsed:
            return max(parsed.items(), key=lambda item: item[1])[0]
    except Exception:
        pass
    return text


def record_limit(rows: Iterable[Any], limit: int | None) -> Iterable[Any]:
    if limit is None:
        return rows
    from itertools import islice

    return islice(rows, limit)


def mirror_oral(limit: int | None = None) -> dict[str, Any]:
    spec = SPECS["oral-images"]
    dataset_slug = slug(spec.name)
    raw_root, train_path = wipe_output(dataset_slug)
    cache_root = Path(
        snapshot_download(
            spec.source,
            repo_type="dataset",
            local_dir=str(CACHE_DIR / dataset_slug),
            local_dir_use_symlinks=False,
        )
    )
    count = 0
    for class_dir in sorted([p for p in cache_root.iterdir() if p.is_dir()]):
        for img_path in sorted(class_dir.glob("*")):
            if limit is not None and count >= limit:
                break
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            rel = f"{class_dir.name}/{img_path.stem}{img_path.suffix.lower()}"
            dest = raw_root / rel
            save_image_file(img_path, dest)
            append_jsonl(
                train_path,
                {
                    "id": f"{dataset_slug}_{count:06d}",
                    "source_dataset": spec.source,
                    "kind": spec.kind,
                    "image": str(dest.relative_to(ROOT)),
                    "prompt": spec.prompt,
                    "response": class_dir.name,
                },
            )
            count += 1
    return {
        "dataset": spec.name,
        "source": spec.source,
        "kind": spec.kind,
        "rows_written": count,
        "train_jsonl": str(train_path.relative_to(ROOT)),
        "notes": "mirror of class-folder oral lesion images",
    }


def mirror_pad_ufes(limit: int | None = None) -> dict[str, Any]:
    spec = SPECS["PAD-UFES-20"]
    dataset_slug = slug(spec.name)
    raw_root, train_path = wipe_output(dataset_slug)
    repo_files = list_repo_files(spec.source, repo_type="dataset")
    image_files = [f for f in repo_files if f.startswith("all_images/") and f.lower().endswith(".png")]
    meta_path = Path(hf_hub_download(spec.source, "metadata.csv", repo_type="dataset"))
    meta = pd.read_csv(meta_path)
    selected_names = meta["img_id"].tolist() if limit is None else meta["img_id"].tolist()[:limit]
    selected_files = []
    image_lookup = {Path(f).name: f for f in image_files}
    for name in selected_names:
        if name in image_lookup:
            selected_files.append(image_lookup[name])
    cache_root = Path(
        snapshot_download(
            spec.source,
            repo_type="dataset",
            local_dir=str(CACHE_DIR / dataset_slug),
            local_dir_use_symlinks=False,
            allow_patterns=["metadata.csv", *selected_files],
        )
    )
    meta = pd.read_csv(cache_root / "metadata.csv")
    image_index = {p.name: p for p in cache_root.joinpath("all_images").rglob("*") if p.is_file()}
    count = 0
    for _, row in meta.iterrows():
        if limit is not None and count >= limit:
            break
        img_name = str(row["img_id"])
        src = image_index.get(img_name)
        if src is None:
            continue
        diagnosis = safe_str(row.get("diagnostic"))
        region = safe_str(row.get("region"))
        symptoms = []
        for field in ["itch", "grew", "hurt", "changed", "bleed", "elevation"]:
            if boolish(row.get(field)):
                symptoms.append(field)
        age = safe_str(row.get("age"))
        sex = safe_str(row.get("gender"))
        fitz = safe_str(row.get("fitspatrick"))
        lesion_id = safe_str(row.get("lesion_id"))
        dest = raw_root / "images" / img_name
        save_image_file(src, dest)
        prompt = spec.prompt
        if region:
            prompt = f"{prompt} The lesion is on the {region.lower()}."
        if symptoms:
            prompt += f" The patient reports: {', '.join(symptoms)}."
        if age or sex or fitz:
            prompt += f" Patient context: age {age or 'unknown'}, sex {sex or 'unknown'}, fitzpatrick {fitz or 'unknown'}."
        append_jsonl(
            train_path,
            {
                "id": f"{dataset_slug}_{count:06d}",
                "source_dataset": spec.source,
                "kind": spec.kind,
                "image": str(dest.relative_to(ROOT)),
                "prompt": prompt,
                "response": diagnosis,
                "metadata": {
                    "patient_id": safe_str(row.get("patient_id")),
                    "lesion_id": lesion_id,
                    "region": region,
                    "age": age,
                    "gender": sex,
                    "fitspatrick": fitz,
                    "diameter_1": safe_str(row.get("diameter_1")),
                    "diameter_2": safe_str(row.get("diameter_2")),
                },
            },
        )
        count += 1
    return {
        "dataset": spec.name,
        "source": spec.source,
        "kind": spec.kind,
        "rows_written": count,
        "train_jsonl": str(train_path.relative_to(ROOT)),
        "notes": "smartphone skin lesions with metadata",
    }


def mirror_fluo_sc(limit: int | None = None) -> dict[str, Any]:
    spec = SPECS["FLUO-SC"]
    dataset_slug = slug(spec.name)
    raw_root, train_path = wipe_output(dataset_slug)
    cache_root = Path(
        snapshot_download(
            spec.source,
            repo_type="dataset",
            local_dir=str(CACHE_DIR / dataset_slug),
            local_dir_use_symlinks=False,
            allow_patterns=["data/CLI.zip", "data/FLUO.zip", "README.md", "Description.txt"],
        )
    )
    extracted_root = cache_root / "_extracted"
    extracted_root.mkdir(exist_ok=True)
    for zip_name in ["CLI.zip", "FLUO.zip"]:
        zip_path = cache_root / "data" / zip_name
        if zip_path.exists():
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(extracted_root)
    count = 0
    for modality_dir in sorted([p for p in extracted_root.rglob("*") if p.is_dir() and p.name in {"CLI", "FLUO"}]):
        modality = modality_dir.name
        for img_path in sorted(modality_dir.rglob("*")):
            if limit is not None and count >= limit:
                break
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            parts = img_path.parts
            label = next((p for p in parts if p in {"BCC", "SCC", "MEL", "ACK", "SEK", "NEV"}), "")
            if not label:
                continue
            dest = raw_root / modality / label / img_path.name
            save_image_file(img_path, dest)
            prompt = "Look at the smartphone skin-lesion photo and provide the most likely diagnosis."
            if modality == "FLUO":
                prompt = "Look at the fluorescence companion image of the skin lesion and provide the most likely diagnosis."
            append_jsonl(
                train_path,
                {
                    "id": f"{dataset_slug}_{count:06d}",
                    "source_dataset": spec.source,
                    "kind": spec.kind,
                    "image": str(dest.relative_to(ROOT)),
                    "prompt": prompt,
                    "response": label,
                    "metadata": {"modality": modality},
                },
            )
            count += 1
    return {
        "dataset": spec.name,
        "source": spec.source,
        "kind": spec.kind,
        "rows_written": count,
        "train_jsonl": str(train_path.relative_to(ROOT)),
        "notes": "CLI/FLUO zip mirror from Mendeley-backed HF repo",
    }


def mirror_scin(limit: int | None = None) -> dict[str, Any]:
    spec = SPECS["SCIN"]
    dataset_slug = slug(spec.name)
    raw_root, train_path = wipe_output(dataset_slug)
    repo_files = list_repo_files(spec.source, repo_type="dataset")
    parquet_files = sorted([f for f in repo_files if f.startswith("data/") and f.lower().endswith(".parquet")])
    if limit is None:
        selected_parquets = parquet_files
    else:
        n_files = max(1, math.ceil(limit / 150))
        selected_parquets = parquet_files[:n_files]
    cache_root = Path(
        snapshot_download(
            spec.source,
            repo_type="dataset",
            local_dir=str(CACHE_DIR / dataset_slug),
            local_dir_use_symlinks=False,
            allow_patterns=["README.md", *selected_parquets],
        )
    )
    parquet_files = sorted((cache_root / "data").glob("*.parquet"))
    count = 0
    for pq_path in parquet_files:
        if limit is not None and count >= limit:
            break
        table = pq.read_table(pq_path)
        for row in table.to_pylist():
            if limit is not None and count >= limit:
                break
            case_id = safe_str(row.get("case_id")) or f"case_{count:06d}"
            label = parse_weighted_label(row.get("weighted_skin_condition_label")) or first_nonempty(
                row.get("dermatologist_skin_condition_on_label_name"),
                row.get("related_category"),
            )
            if not label:
                label = "unknown"

            body_parts = [
                part.replace("body_parts_", "").replace("_", " ")
                for part in row.keys()
                if str(part).startswith("body_parts_") and boolish(row.get(part))
            ]
            symptoms = [
                part.replace("condition_symptoms_", "").replace("other_symptoms_", "").replace("_", " ")
                for part in row.keys()
                if (str(part).startswith("condition_symptoms_") or str(part).startswith("other_symptoms_"))
                and boolish(row.get(part))
            ]
            prompt_bits = [spec.prompt]
            if body_parts:
                prompt_bits.append(f"The affected body part appears to be: {', '.join(body_parts)}.")
            if symptoms:
                prompt_bits.append(f"Reported symptoms include: {', '.join(symptoms)}.")
            shot_types = [
                safe_str(row.get("image_1_shot_type")),
                safe_str(row.get("image_2_shot_type")),
                safe_str(row.get("image_3_shot_type")),
            ]
            prompt_bits = [bit for bit in prompt_bits if bit]
            images = []
            for i in (1, 2, 3):
                field = row.get(f"image_{i}_path")
                if isinstance(field, dict) and field.get("bytes"):
                    images.append((i, field["bytes"]))

            for img_idx, img_bytes in images:
                if limit is not None and count >= limit:
                    break
                dest = raw_root / case_id / f"image_{img_idx}.png"
                save_image_bytes(img_bytes, dest)
                append_jsonl(
                    train_path,
                    {
                        "id": f"{dataset_slug}_{count:06d}",
                        "source_dataset": spec.source,
                        "kind": spec.kind,
                        "image": str(dest.relative_to(ROOT)),
                        "prompt": " ".join(prompt_bits),
                        "response": label,
                        "metadata": {
                            "case_id": case_id,
                            "year": safe_str(row.get("year")),
                            "age_group": safe_str(row.get("age_group")),
                            "sex_at_birth": safe_str(row.get("sex_at_birth")),
                            "fitzpatrick_skin_type": safe_str(row.get("fitzpatrick_skin_type")),
                            "related_category": safe_str(row.get("related_category")),
                            "condition_duration": safe_str(row.get("condition_duration")),
                            "shot_type": safe_str(row.get(f"image_{img_idx}_shot_type")),
                            "weighted_skin_condition_label": safe_str(row.get("weighted_skin_condition_label")),
                            "dermatologist_skin_condition_on_label_name": safe_str(row.get("dermatologist_skin_condition_on_label_name")),
                        },
                    },
                )
                count += 1
    return {
        "dataset": spec.name,
        "source": spec.source,
        "kind": spec.kind,
        "rows_written": count,
        "train_jsonl": str(train_path.relative_to(ROOT)),
        "notes": "consumer dermatology parquet with embedded image bytes",
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build the ViVLM curated visual medical corpus.")
    parser.add_argument("--datasets", nargs="*", default=["oral-images", "PAD-UFES-20", "FLUO-SC", "SCIN"])
    parser.add_argument("--limit-scin", type=int, default=1000, help="Cap SCIN image records to avoid overgrowth.")
    parser.add_argument("--limit-oral", type=int, default=None)
    parser.add_argument("--limit-pad", type=int, default=None)
    parser.add_argument("--limit-fluo", type=int, default=None)
    args = parser.parse_args()

    ensure_dirs()
    manifests: list[dict[str, Any]] = []

    for dataset in args.datasets:
        print(f"[ViVLM] mirroring {dataset}")
        if dataset == "oral-images":
            manifests.append(mirror_oral(args.limit_oral))
        elif dataset == "PAD-UFES-20":
            manifests.append(mirror_pad_ufes(args.limit_pad))
        elif dataset == "FLUO-SC":
            manifests.append(mirror_fluo_sc(args.limit_fluo))
        elif dataset == "SCIN":
            manifests.append(mirror_scin(args.limit_scin))
        else:
            raise SystemExit(f"Unknown dataset: {dataset}")

    write_manifest(manifests)
    print("[ViVLM] wrote manifest:", MANIFEST_DIR / "downloaded_datasets.csv")


if __name__ == "__main__":
    main()
