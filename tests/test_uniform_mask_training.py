import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "trainpadspecmodel_uniform_mask",
    ROOT / "trainpadspecmodel_uniform_mask.py",
)
uniform_mask = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(uniform_mask)


def test_uniform_mask_policy_does_not_depend_on_epoch():
    spectra = torch.ones((4, 43, 50), dtype=torch.float32)
    generator_early = torch.Generator().manual_seed(781)
    generator_late = torch.Generator().manual_seed(781)

    early, early_usable = uniform_mask.build_uniform_mask_batch(
        spectra,
        epoch=0,
        batch_index=5,
        generator=generator_early,
    )
    late, late_usable = uniform_mask.build_uniform_mask_batch(
        spectra,
        epoch=9999,
        batch_index=5,
        generator=generator_late,
    )

    assert torch.equal(early, late)
    assert torch.equal(early_usable, late_usable)


def test_extra_block_is_scheduled_every_fifth_batch():
    assert uniform_mask.EXTRA_BLOCK_INTERVAL == 5
    assert uniform_mask.EXTRA_BLOCK_MIN_BANDS == 1
    assert uniform_mask.EXTRA_BLOCK_MAX_BANDS == 12
    assert uniform_mask.OPPOSITE_EDGE_PROBABILITY == 0.5
    assert uniform_mask.OPPOSITE_EDGE_MIN_BANDS == 1
    assert uniform_mask.OPPOSITE_EDGE_MAX_BANDS == 12
    assert uniform_mask.LARGE_BAND_LIMIT == 30


def test_extra_block_can_be_one_50_nm_band():
    generator = torch.Generator().manual_seed(20260915)
    widths = [
        end - start
        for start, end in (
            uniform_mask.sample_extra_block(generator) for _ in range(1000)
        )
    ]
    assert min(widths) == 1
    assert max(widths) == 12


def test_launcher_replaces_only_the_augmentation_sampler():
    assert (
        uniform_mask.training.build_augmented_batch
        is uniform_mask.build_uniform_mask_batch
    )
    assert (
        uniform_mask.variant_defaults["flexible_pure_dense_ffn"][
            "context_encoding"
        ]
        == "word_split_v2"
    )
