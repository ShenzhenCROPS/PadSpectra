import pytest
import torch

import trainpadspecmodel_new as training


def test_spectral_loss_reports_reconstruction_separately_from_regularizers():
    criterion = training.MAELoss(bandnum=2, bandwidth=3)
    target = torch.ones((1, 2, 3), dtype=torch.float32)
    prediction = target.clone()
    prediction[:, 0] = 0.75
    missing_band_mask = torch.tensor([[[True], [False]]])

    components = criterion.components(target, prediction, missing_band_mask)

    torch.testing.assert_close(
        components["reconstruction_mae"],
        torch.tensor(0.25),
    )
    torch.testing.assert_close(
        components["reconstruction_absolute_error_sum"],
        torch.tensor(0.75),
    )
    torch.testing.assert_close(
        components["reconstruction_squared_error_sum"],
        torch.tensor(0.1875),
    )
    assert components["reconstruction_points"].item() == 3
    torch.testing.assert_close(
        components["objective"],
        components["reconstruction_mae"]
        + 2 * components["tv"]
        + components["negative"],
    )


def test_spectral_loss_rejects_masked_input_as_reconstruction_target():
    criterion = training.MAELoss(bandnum=2, bandwidth=3)
    masked_input = torch.ones((1, 2, 3), dtype=torch.float32)
    masked_input[:, 0] = 0
    prediction = torch.full_like(masked_input, 0.5)
    missing_band_mask = torch.tensor([[[True], [False]]])

    with pytest.raises(RuntimeError, match="Reconstruction mask contains zero points"):
        criterion(masked_input, prediction, missing_band_mask)


def test_spectral_loss_runs_in_fp32_and_backpropagates_reconstruction_error():
    criterion = training.MAELoss(bandnum=2, bandwidth=3)
    target = torch.ones((1, 2, 3), dtype=torch.float16)
    prediction = torch.full_like(target, 0.5).requires_grad_(True)
    missing_band_mask = torch.tensor([[[True], [False]]])

    components = criterion.components(target, prediction, missing_band_mask)
    components["objective"].backward()

    assert components["objective"].dtype == torch.float32
    assert prediction.grad is not None
    assert prediction.grad[:, 0].abs().sum().item() > 0


def test_large_fp16_reconstruction_mask_does_not_overflow_denominator():
    """Regression test for the GPU-batch bug that reduced reconstruction to zero."""
    criterion = training.MAELoss()
    target = torch.ones((40, 43, 50), dtype=torch.float16)
    prediction = torch.full_like(target, 0.5)
    missing_band_mask = torch.ones((40, 43, 1), dtype=torch.bool)

    components = criterion.components(target, prediction, missing_band_mask)

    assert components["reconstruction_points"].item() == 86_000
    torch.testing.assert_close(
        components["reconstruction_mae"],
        torch.tensor(0.5),
    )
