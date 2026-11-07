# PadSpectra

PadSpectra completes missing wavelength regions in heterogeneous leaf and soil
spectra. This repository provides the dense publication model and the code to
train it and reconstruct spectra.

This public repository contains one model: `flexible_pure_dense_ffn`, the dense
feed-forward PadSpectra configuration used for the final results. It is trained
for 10,000 epochs with the uniform masking policy in
`trainpadspecmodel_uniform_mask.py`.

## Repository contents

- `padspecmodels_new.py`: model architecture and inference-time padding
- `trainpadspecmodel_new.py`: data loading, conditioning, losses, validation,
  EMA, checkpointing, and resumption
- `trainpadspecmodel_uniform_mask.py`: publication masking policy
- `scripts/train_publication_model.sh`: publication training launcher
- `infer_padspectra.py`: checkpoint loading and batch spectral reconstruction
- `tests/`: model and training unit tests
- `docs/`: architecture and training details

A small de-identified Gemini spectra example is provided under `Data/examples/`.
The complete datasets, trained weights, generated predictions, and intermediate
files are not stored in Git. No pretrained checkpoint is currently published;
the trained weights are planned for release after acceptance of the associated
paper. Expected data and checkpoint layouts are documented in `Data/README.md`
and `weights/README.md`.

## Installation

```bash
git clone https://github.com/ShenzhenCROPS/PadSpectra.git
cd PadSpectra
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For development and tests:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## Train the publication model

Place the partitioned data under `Data/`, or set `PADSPECTRA_DATA_DIR` to an
equivalent directory. Then run:

```bash
PADSPECTRA_DATA_DIR=/path/to/padspectra-data \
  bash scripts/train_publication_model.sh
```

The launcher selects:

- variant: `flexible_pure_dense_ffn`
- context encoding: `word_split_v2`
- epochs: 10,000
- learning-rate plateau end: epoch 2,000
- primary cosine schedule end: epoch 9,000
- random seed: `20260804`
- EMA decay: `0.99`, updated every five successful optimizer steps

Training outputs are written below `runs/` by default. Paths, the run name, and
the training schedule can be overridden through the documented
`PADSPECTRA_*` environment variables; the model architecture remains fixed.

## Reconstruct spectra

```bash
python infer_padspectra.py \
  --checkpoint /path/to/training_checkpoint_180_best_validation.pth \
  --spectra /path/to/partial_spectra.h5 \
  --metadata /path/to/metadata.json \
  --output outputs/completed_spectra.h5
```

The input HDF5 file has a two-dimensional `spectra` dataset with one spectrum
per row and a one-dimensional `wavelength` dataset in nm. Use a consecutive
1 nm grid from 350 to 2499 nm, with 2500 nm optional, and mark missing values
with zeros. The output uses the same grid and preserves measured values. If
the input includes a missing 2500 nm value, inference extrapolates it from
2495–2499 nm.

Supply one metadata list per spectrum, in the same order. For example, a
file for one spectrum could contain:

```json
[["Sample : Leaf", "Spectral type : Reflectance"]]
```

Inference uses the checkpoint's EMA weights, vocabulary, and numerical
normalization statistics. When using a standalone EMA snapshot, also pass the
matching `training_metrics_*.json` file with `--training-metrics`. A checkpoint
from your own training run is required until pretrained weights are released.

## Reproducibility and data use

Use the predefined train, validation, and test indices without resampling.
Obtain source spectra under their original licenses.

## License

The source code is provided under the Apache License 2.0. Source datasets have
their own licenses.
