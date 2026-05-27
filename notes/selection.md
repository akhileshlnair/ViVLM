# Dataset selection notes

We are prioritizing datasets that look like real user-captured photos of body parts rather than studio-style medical imagery.

## Included first pass

- PAD-UFES-20: smartphone skin lesions
- SCIN: consumer dermatology images with clinical labels
- FLUO-SC: smartphone-captured skin lesions with paired fluorescence
- oral-images: oral cavity lesions

## Why these

These datasets are the closest public fit for a model that needs to inspect photos from a phone and help with diagnosis or triage.

## What we are avoiding for now

- purely synthetic medical images
- lab-only imagery without real-world capture conditions
- datasets that are mainly text, not images
- sources with access restrictions that make the corpus brittle
