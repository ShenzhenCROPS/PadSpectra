#!/usr/bin/env python3
"""Train the published dense PadSpectra model with one mask policy for all epochs.

This launcher preserves ``trainpadspecmodel_new.py`` and its historical mask
schedule.  It replaces only the spectral augmentation sampler:

* one large edge mask of 1--29 complete 50 nm bands;
* a short mask of 1--12 bands at the opposite edge with 50% probability;
* one additional contiguous mask of 1--12 bands (50--600 nm) every fifth
  batch.

The same distribution is used from the first through the last epoch.
"""

import os

import torch


training_variant = os.environ.setdefault(
    "PADSPECTRA_TRAINING_VARIANT",
    "flexible_pure_dense_ffn",
)
variant_defaults = {
    "flexible_pure_dense_ffn": {
        "context_encoding": "word_split_v2",
        "run_name": (
            "flexible_pure_dense_ffn_word_split_v2_uniform_mask12x12_10k"
        ),
    },
}
if training_variant not in variant_defaults:
    supported = ", ".join(sorted(variant_defaults))
    raise ValueError(
        f"Uniform-mask training supports {supported}; got {training_variant!r}"
    )

variant_default = variant_defaults[training_variant]
os.environ.setdefault(
    "PADSPECTRA_CONTEXT_ENCODING",
    variant_default["context_encoding"],
)
os.environ.setdefault("PADSPECTRA_RUN_NAME", variant_default["run_name"])

import trainpadspecmodel_new as training  # noqa: E402


LARGE_BAND_LIMIT = 30  # torch.randint upper bound: 1--29 bands
OPPOSITE_EDGE_PROBABILITY = 0.5
OPPOSITE_EDGE_MIN_BANDS = 1
OPPOSITE_EDGE_MAX_BANDS = 12
EXTRA_BLOCK_INTERVAL = 5
EXTRA_BLOCK_MIN_BANDS = 1
EXTRA_BLOCK_MAX_BANDS = 12


def sample_extra_block(generator=None):
    """Sample an extra block spanning one to twelve complete 50 nm bands."""
    block_start = int(
        torch.randint(
            0,
            training.bandnum - 3,
            (),
            generator=generator,
        ).item()
    )
    block_bands = int(
        torch.randint(
            EXTRA_BLOCK_MIN_BANDS,
            EXTRA_BLOCK_MAX_BANDS + 1,
            (),
            generator=generator,
        ).item()
    )
    block_end = min(training.bandnum, block_start + block_bands)
    return block_start, block_end


def build_uniform_mask_batch(
    spectra_batch,
    epoch,
    batch_index,
    generator=None,
):
    """Apply the epoch-independent three-region augmentation policy."""
    del epoch
    all0real = (spectra_batch == 0).sum()
    non_zero_mask = torch.ones(
        spectra_batch.size(0),
        dtype=torch.bool,
        device=spectra_batch.device,
    )
    spectra_batch_input = spectra_batch

    for _ in range(16):
        spectra_batch_input = spectra_batch.clone()
        large_bands = int(
            torch.randint(
                1,
                LARGE_BAND_LIMIT,
                (),
                generator=generator,
            ).item()
        )
        mask_opposite_edge = bool(
            torch.rand((), generator=generator).item()
            < OPPOSITE_EDGE_PROBABILITY
        )

        if batch_index % 2 == 0:
            spectra_batch_input[:, :large_bands, :] = 0
            if mask_opposite_edge:
                short_bands = int(
                    torch.randint(
                        OPPOSITE_EDGE_MIN_BANDS,
                        OPPOSITE_EDGE_MAX_BANDS + 1,
                        (),
                        generator=generator,
                    ).item()
                )
                spectra_batch_input[:, -short_bands:, :] = 0
        else:
            spectra_batch_input[:, -large_bands:, :] = 0
            if mask_opposite_edge:
                short_bands = int(
                    torch.randint(
                        OPPOSITE_EDGE_MIN_BANDS,
                        OPPOSITE_EDGE_MAX_BANDS + 1,
                        (),
                        generator=generator,
                    ).item()
                )
                spectra_batch_input[:, :short_bands, :] = 0

        if batch_index % EXTRA_BLOCK_INTERVAL == 0:
            block_start, block_end = sample_extra_block(generator)
            spectra_batch_input[:, block_start:block_end, :] = 0

        zero_mask_input = spectra_batch_input == 0
        non_zero_mask = training.usable_spectrum_mask(spectra_batch_input)
        all0inbatch = zero_mask_input.sum()
        if non_zero_mask.any() and all0inbatch > all0real:
            break

    return spectra_batch_input, non_zero_mask


training.build_augmented_batch = build_uniform_mask_batch


if __name__ == "__main__":
    training.main()
