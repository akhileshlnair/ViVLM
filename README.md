# ViVLM

ViVLM is a curated visual medical corpus for training models that help users photograph body parts and get a more accurate diagnosis.

The repo focuses on high-signal, public datasets where images come from smartphones, portable devices, or real-world consumer captures.

## Current curation targets

- PAD-UFES-20: smartphone skin-lesion images with metadata
- SCIN: consumer-contributed dermatology images with dermatologist labels
- FLUO-SC: skin-lesion images collected from smartphones
- oral-images: oral lesion images
- PMC-VQA: large medical VQA pairs from PubMed Central with reasoning traces
- MIMIC-CXR-VQA: chest radiograph VQA with question-answer supervision

## Repo layout

- `data/raw/` stores the mirrored image files
- `data/train/` stores normalized trainable JSONL records
- `data/manifests/` tracks source metadata and download status
- `scripts/` contains download and normalization helpers

## What this repo is for

The goal is to turn public body-part image datasets into a clean multimodal training set with:

- image paths
- diagnosis labels or caption targets
- useful metadata for supervised fine-tuning

The first pass is intentionally conservative: only datasets that are actually useful for real-world phone-style diagnosis work make the cut.
