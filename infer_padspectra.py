"""Reconstruct missing wavelengths in row-aligned PadSpectra inputs."""

import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np
import torch

from padspecmodels_new import AutoEncoder, validate_dense_state_dict
import trainpadspecmodel_new as training


FIT_2500_WEIGHTS = np.array([-0.4, -0.1, 0.2, 0.5, 0.8], dtype=np.float32)


def load_model(checkpoint_path, training_metrics, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "ema_model_state_dict" in checkpoint:
        if checkpoint.get("training_variant") not in training.TRAINING_VARIANTS:
            raise ValueError("Checkpoint is not from the published PadSpectra model")
        state_dict = checkpoint["ema_model_state_dict"]
        context = checkpoint["context"]
    else:
        if training_metrics is None:
            raise ValueError("A standalone EMA snapshot requires --training-metrics")
        state_dict = checkpoint
        with training_metrics.open(encoding="utf-8") as handle:
            context = json.load(handle)["context"]

    encoding = context.get("context_encoding", training.ATOMIC_CONTEXT_ENCODING)
    validate_dense_state_dict(state_dict)
    model = AutoEncoder(
        headnum=10,
        vocab_size=context["vocab_size"],
        pad_index=context["pad_index"],
        bandwidth=training.bandwidth,
        bandnum=training.bandnum,
        maxwords=training.CONTEXT_TOKEN_COUNTS[encoding],
        hidden_dim=training.hidden_dim,
    )
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval(), context


def reconstruct_file(args, model, context, device):
    with args.metadata.open(encoding="utf-8") as handle:
        metadata_rows = json.load(handle)
    if not isinstance(metadata_rows, list) or not all(
        isinstance(row, list) for row in metadata_rows
    ):
        raise ValueError("Metadata must be a JSON list of row lists")

    with h5py.File(args.spectra, "r") as source:
        spectra = source["spectra"]
        wavelengths = source["wavelength"][:]
        modeled_count = training.bandnum * training.bandwidth
        valid_lengths = (modeled_count, modeled_count + 1)
        if spectra.ndim != 2 or spectra.shape[1] not in valid_lengths:
            raise ValueError("Expected spectra shaped (rows, 2150) or (rows, 2151)")
        if not np.array_equal(wavelengths, np.arange(350, 350 + spectra.shape[1])):
            raise ValueError("Expected a 1 nm wavelength grid from 350 to 2499 or 2500")
        if len(spectra) == 0 or len(spectra) != len(metadata_rows):
            raise ValueError("Spectra and metadata must have the same nonzero row count")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=args.output.parent) as temporary_dir:
            temporary_path = Path(temporary_dir) / "completed.h5"
            with h5py.File(temporary_path, "w") as output, torch.inference_mode():
                output.create_dataset("wavelength", data=wavelengths)
                reconstructed = output.create_dataset(
                    "spectra", spectra.shape, dtype="f4"
                )
                for start in range(0, len(spectra), args.batch_size):
                    stop = min(start + args.batch_size, len(spectra))
                    batch = spectra[start:stop].astype(np.float32, copy=False)
                    if not np.isfinite(batch).all() or (batch < 0).any():
                        raise ValueError(
                            f"Rows {start}:{stop} must be finite and nonnegative"
                        )
                    modeled = batch[:, :modeled_count]
                    shaped = modeled.reshape(-1, training.bandnum, training.bandwidth)
                    inputs = torch.from_numpy(shaped).to(device)
                    encoded = training.encode_runtime_conditioning(
                        metadata_rows[start:stop], context
                    )
                    conditioning = torch.from_numpy(encoded).to(device)
                    with torch.autocast(device.type, enabled=device.type == "cuda"):
                        completed = model.pad(inputs, conditioning)
                    completed = completed.float().cpu().numpy().reshape(modeled.shape)
                    modeled[:] = np.where(modeled > 0, modeled, completed)
                    if spectra.shape[1] == modeled_count + 1:
                        fitted = np.maximum(modeled[:, -5:] @ FIT_2500_WEIGHTS, 0)
                        batch[:, -1] = np.where(batch[:, -1] > 0, batch[:, -1], fitted)
                    if not np.isfinite(batch).all():
                        raise ValueError(
                            f"Reconstruction produced non-finite values in rows {start}:{stop}"
                        )
                    reconstructed[start:stop] = batch

            # Publish the finished file without overwriting an existing result.
            os.link(temporary_path, args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--training-metrics", type=Path, help="Required for a standalone EMA snapshot"
    )
    parser.add_argument("--spectra", type=Path, required=True, help="Input HDF5")
    parser.add_argument("--metadata", type=Path, required=True, help="Row-aligned metadata JSON")
    parser.add_argument("--output", type=Path, required=True, help="New HDF5 file")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.output.exists():
        parser.error("--output must point to a new file")
    device = training.device if args.device == "auto" else torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    model, context = load_model(args.checkpoint, args.training_metrics, device)
    reconstruct_file(args, model, context, device)


if __name__ == "__main__":
    main()
