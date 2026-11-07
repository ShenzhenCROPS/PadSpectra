"""Tests for the published dense PadSpectra architecture."""

import unittest

import torch

import padspecmodels_new as psm


def build_model():
    return psm.AutoEncoder(
        headnum=1,
        vocab_size=8,
        pad_index=0,
        bandwidth=4,
        bandnum=3,
        maxwords=2,
        hidden_dim=4,
    )


class TestDenseFeedForward(unittest.TestCase):
    def test_model_uses_dense_feed_forward_layers(self):
        model = build_model()

        self.assertEqual(
            sum(isinstance(m, psm.DenseFeedForward) for m in model.modules()),
            7,
        )

    def test_dense_model_forward_and_backward(self):
        torch.manual_seed(5)
        model = build_model()
        spectra = torch.rand(2, 3, 4)
        spectra[:, 1] = 0
        context = torch.tensor(
            [
                [[1, 2], [3, 0]],
                [[2, 1], [4, 0]],
            ]
        )

        prediction, missing_mask = model(spectra, context)
        prediction.square().mean().backward()

        self.assertEqual(prediction.shape, spectra.shape)
        self.assertEqual(missing_mask.shape, (2, 3, 1))
        self.assertTrue(missing_mask[:, 1].all())
        dense_parameters = [
            parameter
            for module in model.modules()
            if isinstance(module, psm.DenseFeedForward)
            for parameter in module.parameters()
        ]
        self.assertTrue(all(parameter.grad is not None for parameter in dense_parameters))

    def test_dense_checkpoint_is_validated_and_reloadable(self):
        model = build_model()
        self.assertIsNone(psm.validate_dense_state_dict(model.state_dict()))

        reloaded = build_model()
        reloaded.load_state_dict(model.state_dict(), strict=True)


if __name__ == "__main__":
    unittest.main()
