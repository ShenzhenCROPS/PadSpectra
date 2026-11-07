import numpy as np
import torch
import json

import padspecmodels_new as models
import trainpadspecmodel_new as training


def _decoded_numeric(encoded_value):
    return -float(encoded_value) - models.NUMERIC_TOKEN_OFFSET


def test_only_publication_dense_model_is_exposed():
    assert training.TRAINING_VARIANT_NAMES == ("flexible_pure_dense_ffn",)
    config = training.TRAINING_VARIANTS["flexible_pure_dense_ffn"]
    assert config == {
        "conditioning_mode": "flexible_selected_properties",
        "property_constraint": False,
        "spectrometer_dropout": True,
        "measurement_mode_dropout": True,
    }
    assert training.TRAINING_FEED_FORWARD_TYPE == models.FEED_FORWARD_DENSE
    assert training.TRAINING_SEED == 20260804
    assert training.TRAINING_RUN_NAME == training.TRAINING_VARIANT
    assert training.property_loss_weight == 0.0005


def test_extended_learning_rate_schedule_is_continuous_and_decays():
    steps_per_epoch = 48
    boundary_step = (
        training.initial_schedule_end_epoch + 1
    ) * steps_per_epoch
    final_step = (training.num_epochs + 1) * steps_per_epoch

    boundary_factor = training.learning_rate_factor(boundary_step, steps_per_epoch)
    first_continuation_factor = training.learning_rate_factor(
        boundary_step + 1,
        steps_per_epoch,
    )
    final_factor = training.learning_rate_factor(final_step, steps_per_epoch)

    assert training.num_epochs == 25000
    assert abs(boundary_factor - 0.1) < 1e-12
    assert first_continuation_factor <= boundary_factor
    assert abs(first_continuation_factor - boundary_factor) < 1e-9
    assert abs(
        final_factor
        - training.continuation_final_learning_rate / training.learningRate
    ) < 1e-12


def test_hybrid_10k_learning_rate_schedule_boundaries(monkeypatch):
    monkeypatch.setattr(training, "num_epochs", 10000)
    monkeypatch.setattr(training, "lr_plateau_end_epoch", 2000)
    monkeypatch.setattr(training, "initial_schedule_end_epoch", 9000)
    steps_per_epoch = 48

    plateau_end_step = (training.lr_plateau_end_epoch + 1) * steps_per_epoch
    primary_end_step = (
        training.initial_schedule_end_epoch + 1
    ) * steps_per_epoch
    final_step = (training.num_epochs + 1) * steps_per_epoch
    primary_midpoint_step = (
        plateau_end_step + primary_end_step
    ) // 2

    assert training.learning_rate_factor(250, steps_per_epoch) == 0.5
    assert training.learning_rate_factor(500, steps_per_epoch) == 1.0
    assert training.learning_rate_factor(plateau_end_step, steps_per_epoch) == 1.0
    assert abs(
        training.learning_rate_factor(primary_midpoint_step, steps_per_epoch)
        - 0.55
    ) < 1e-12
    assert abs(
        training.learning_rate_factor(primary_end_step, steps_per_epoch) - 0.1
    ) < 1e-12
    assert abs(
        training.learning_rate_factor(final_step, steps_per_epoch)
        - training.continuation_final_learning_rate / training.learningRate
    ) < 1e-12


def test_ema_uses_requested_decay_and_five_step_update_frequency():
    assert training.ema_decay == 0.99
    assert training.ema_update_interval == 5
    assert [
        step for step in range(1, 16) if training.should_update_ema(step)
    ] == [5, 10, 15]
    effective_per_step_decay = (
        training.ema_decay ** (1.0 / training.ema_update_interval)
    )
    assert abs(effective_per_step_decay - 0.997992) < 1e-6
    assert training.ema_checkpoint_interval == 1000
    assert [
        epoch
        for epoch in range(training.num_epochs + 1)
        if training.should_save_ema_snapshot(epoch)
    ] == list(range(1000, 25001, 1000))


def test_conditioning_schedule_requires_one_run_level_mode():
    encoded = torch.full(
        (2, training.max_context_entries, training.max_context_tokens),
        7.0,
    )
    try:
        training.apply_conditioning_schedule(encoded, pad_index=7)
    except ValueError as error:
        assert "fixed conditioning mode" in str(error)
    else:
        raise AssertionError("Conditioning unexpectedly mixed modes by row")


def test_eight_complete_bands_are_usable_but_seven_are_not():
    spectra = torch.ones(
        2,
        training.bandnum,
        training.bandwidth,
    )
    spectra[0, : training.maximum_missing_bands] = 0
    spectra[1, : training.maximum_missing_bands + 1] = 0

    usable = training.usable_spectrum_mask(spectra)

    assert training.minimum_observed_bands == 8
    assert training.maximum_missing_bands == 35
    assert usable.tolist() == [True, False]


def test_predefined_index_file_is_authoritative(tmp_path):
    rows = [
        [f"Dataset : D{index % 2}"]
        for index in range(6)
    ]
    partitions = {
        "train": [4, 0, 3],
        "validation": [1],
        "test": [5, 2],
    }
    payload = {
        "row_count": len(rows),
        "processing_profile": "unit-test",
        "index_semantics": "zero-based",
        "partitions": {},
    }
    for name, indices in partitions.items():
        counts = {}
        for index in indices:
            dataset = f"D{index % 2}"
            counts[dataset] = counts.get(dataset, 0) + 1
        payload["partitions"][name] = {
            "rows": len(indices),
            "indices": indices,
            "dataset_labels": sorted(counts),
            "dataset_row_counts": counts,
        }
    split_path = tmp_path / "split_indices.json"
    split_path.write_text(json.dumps(payload))

    train, validation, metadata = training.load_predefined_split(
        rows,
        split_path,
        len(rows),
        fraction=0.5,
        seed=999,
    )

    assert train.tolist() == partitions["train"]
    assert validation.tolist() == partitions["validation"]
    assert metadata["strategy"] == "predefined_index_file"
    assert metadata["test_rows"] == 2


def test_runtime_conditioning_parses_only_applicable_selected_properties():
    rows = [
        [
            "Dataset : soil",
            "Sample : Soil",
            "Spectral type : Reflectance",
            "Spectrometer : ASD",
            "pH : 6 | Water",
            "pH : 4 | CaCl2 solution",
            "pH : 5.5",
            "pH : 5 | KCl solution",
            "Water content : 99 g/m2",
            "Water content : 10 %",
        ],
        [
            "Dataset : soil",
            "Sample : Soil",
            "Spectral type : Reflectance",
            "Spectrometer : FOSS",
            "pH : 8 | Water",
            "pH : 6 | CaCl2 solution",
            "pH : 7.5",
            "Water content : 20 %",
        ],
        [
            "Dataset : leaf",
            "Sample : Leaf",
            "Spectral type : Transmittance",
            "Spectrometer : ASD",
            "Measurement mode : Leaf clip",
            "Water content : 100 g/m2",
            "Leaf mass per area : 50 g/m2",
            "pH : 7 | Water",
        ],
        [
            "Dataset : leaf",
            "Sample : Leaf",
            "Spectral type : Reflectance",
            "Spectrometer : FOSS",
            "Measurement mode : Integrating sphere",
            "Water content : 300 g/m2",
            "Leaf mass per area : 150 g/m2",
        ],
    ]

    encoded, metadata = training.build_runtime_conditioning(
        rows,
        np.arange(len(rows)),
    )
    pad = metadata["pad_index"]
    available = encoded[:, :, 0] != pad

    assert encoded.shape == (4, 10, 6)
    assert available[0].tolist() == [
        True, True, True, False, True, True, True, False, False, True
    ]
    assert available[2].tolist() == [
        True, True, True, True, False, False, False, True, True, False
    ]
    assert metadata["numerical_statistics"]["ph_in_water"]["count"] == 2
    assert metadata["numerical_statistics"]["ph_in_cacl2"]["count"] == 2
    assert metadata["numerical_statistics"]["ph_generic"]["count"] == 2
    assert metadata["numerical_statistics"]["leaf_water"]["count"] == 2
    assert metadata["numerical_statistics"]["leaf_lma"]["count"] == 2
    assert metadata["numerical_statistics"]["soil_gravimetric_water"]["count"] == 2

    assert _decoded_numeric(encoded[0, 4, 1]) == -1.0
    assert _decoded_numeric(encoded[1, 4, 1]) == 1.0
    assert _decoded_numeric(encoded[0, 5, 1]) == -1.0
    assert _decoded_numeric(encoded[1, 5, 1]) == 1.0
    assert _decoded_numeric(encoded[0, 6, 1]) == -1.0
    assert _decoded_numeric(encoded[1, 6, 1]) == 1.0
    assert _decoded_numeric(encoded[2, 7, 2]) == -1.0
    assert _decoded_numeric(encoded[3, 7, 2]) == 1.0
    assert _decoded_numeric(encoded[0, 9, 2]) == -1.0
    assert _decoded_numeric(encoded[1, 9, 2]) == 1.0


def test_word_split_v2_uses_shared_words_and_standardized_property_values():
    rows = [
        [
            "Dataset : soil",
            "Sample : Soil",
            "Spectral type : Reflectance",
            "Spectrometer : Thermo Scientific",
            "Measurement mode : Integrating sphere",
            "pH : 6 | Water",
            "pH : 4 | CaCl2 solution",
            "Water content : 10 %",
        ],
        [
            "Dataset : soil",
            "Sample : Soil",
            "Spectral type : Reflectance",
            "Spectrometer : ASD",
            "Measurement mode : Bare fiber",
            "pH : 8 | Water",
            "pH : 6 | CaCl2 solution",
            "Water content : 20 %",
        ],
        [
            "Dataset : leaf",
            "Sample : Leaf",
            "Spectral type : Transmittance",
            "Spectrometer : Thermo Scientific",
            "Measurement mode : Leaf clip",
            "Water content : 100 g/m2",
            "Leaf mass per area : 50 g/m2",
        ],
        [
            "Dataset : leaf",
            "Sample : Leaf",
            "Spectral type : Reflectance",
            "Spectrometer : ASD",
            "Measurement mode : Integrating sphere",
            "Water content : 300 g/m2",
            "Leaf mass per area : 150 g/m2",
        ],
    ]

    encoded, metadata = training.build_runtime_conditioning(
        rows,
        np.arange(len(rows)),
        context_encoding=training.WORD_SPLIT_CONTEXT_ENCODING,
    )
    pad = metadata["pad_index"]

    def decode(entry):
        output = []
        for value in entry:
            if value == pad:
                continue
            if value < -(models.NUMERIC_TOKEN_OFFSET / 2):
                output.append(_decoded_numeric(value))
            else:
                output.append(metadata["idx2word"][str(int(value))])
        return output

    assert encoded.shape == (4, 10, 6)
    assert metadata["context_encoding"] == "word_split_v2"
    assert metadata["max_context_tokens"] == 6
    assert decode(encoded[0, 2]) == ["spectrometer", "thermo", "scientific"]
    assert decode(encoded[0, 3]) == [
        "measurement",
        "mode",
        "integrating",
        "sphere",
    ]
    assert decode(encoded[0, 4]) == ["ph", -1.0, "water"]
    assert decode(encoded[1, 5]) == ["ph", 1.0, "cacl2", "solution"]
    assert decode(encoded[2, 7]) == ["water", "content", -1.0, "g/m2"]
    assert decode(encoded[3, 8]) == [
        "leaf",
        "mass",
        "per",
        "area",
        1.0,
        "g/m2",
    ]


def test_word_split_v2_pads_only_unseen_words_in_a_composite_value():
    rows = [
        [
            "Dataset : train",
            "Sample : Soil",
            "Spectral type : Reflectance",
            "Spectrometer : Thermo Scientific",
        ],
        [
            "Dataset : validation",
            "Sample : Soil",
            "Spectral type : Reflectance",
            "Spectrometer : Novel Scientific",
        ],
    ]
    encoded, metadata = training.build_runtime_conditioning(
        rows,
        np.asarray([0]),
        context_encoding=training.WORD_SPLIT_CONTEXT_ENCODING,
    )
    pad = metadata["pad_index"]
    spectrometer = encoded[1, 2]

    assert metadata["unseen_context_words_padded"] == ["novel"]
    assert metadata["idx2word"][str(int(spectrometer[0]))] == "spectrometer"
    assert spectrometer[1] == pad
    assert metadata["idx2word"][str(int(spectrometer[2]))] == "scientific"


def test_conditioning_modes_and_rich_property_dropout():
    rows = [
        [
            "Dataset : leaf",
            "Sample : Leaf",
            "Spectral type : Reflectance",
            "Spectrometer : ASD",
            "Measurement mode : Leaf clip",
            "Water content : 100 g/m2",
            "Leaf mass per area : 50 g/m2",
        ]
    ] * 6
    encoded, metadata = training.build_runtime_conditioning(
        rows,
        np.arange(len(rows)),
    )
    encoded = torch.from_numpy(encoded)
    modes = torch.arange(6)

    without_optional, returned_modes = training.apply_conditioning_schedule(
        encoded,
        metadata["pad_index"],
        modes=modes,
        spectrometer_dropout_probability=1.0,
        measurement_mode_dropout_probability=1.0,
        numerical_property_retention_probability=0.0,
    )
    with_optional, _ = training.apply_conditioning_schedule(
        encoded,
        metadata["pad_index"],
        modes=modes,
        spectrometer_dropout_probability=0.0,
        measurement_mode_dropout_probability=0.0,
        numerical_property_retention_probability=1.0,
    )

    assert returned_modes.tolist() == modes.tolist()
    assert (without_optional[:, :, 0] != metadata["pad_index"]).sum(1).tolist() == [
        0, 1, 2, 3, 4, 2,
    ]
    assert (with_optional[:, :, 0] != metadata["pad_index"]).sum(1).tolist() == [
        0, 1, 2, 3, 4, 6,
    ]


def test_validation_style_conditioning_uses_all_available_context():
    rows = [
        [
            "Dataset : leaf",
            "Sample : Leaf",
            "Spectral type : Reflectance",
            "Spectrometer : ASD",
            "Measurement mode : Leaf clip",
            "Water content : 100 g/m2",
            "Leaf mass per area : 50 g/m2",
        ]
    ] * 4
    encoded, metadata = training.build_runtime_conditioning(
        rows,
        np.arange(len(rows)),
    )
    encoded = torch.from_numpy(encoded)

    rich_context, _ = training.apply_conditioning_schedule(
        encoded,
        metadata["pad_index"],
        fixed_mode=training.CONDITIONING_MODE_NAMES.index(
            "flexible_selected_properties"
        ),
        mask_optional_context=False,
    )
    fixed_context, _ = training.apply_conditioning_schedule(
        encoded,
        metadata["pad_index"],
        fixed_mode=training.CONDITIONING_MODE_NAMES.index(
            "sample_spectral_spectrometer_and_measurement_mode"
        ),
        fixed_mode_spectrometer_dropout=False,
        fixed_mode_measurement_dropout=False,
        mask_optional_context=False,
    )

    assert (rich_context[:, :, 0] != metadata["pad_index"]).sum(1).tolist() == [
        6, 6, 6, 6,
    ]
    assert (fixed_context[:, :, 0] != metadata["pad_index"]).sum(1).tolist() == [
        4, 4, 4, 4,
    ]
    assert training.best_validation_metric == "masked_reconstruction_mae_v2"
    assert training.validation_conditioning_policy == "all_available_context"


def test_word_embedding_decodes_standardized_numeric_tokens():
    embedding = models.WordEmbedding(vocab_size=4, embed_size=3, pad_index=3)
    with torch.no_grad():
        embedding.vocab_emb.weight.zero_()
        embedding.float_proj.weight.fill_(1.0)
        embedding.float_proj.bias.zero_()

    encoded_number = -2.5 - models.NUMERIC_TOKEN_OFFSET
    output, mask = embedding(torch.tensor([[encoded_number, 3.0]]))

    torch.testing.assert_close(output[0, 0], torch.full((3,), 2.5))
    assert mask.tolist() == [[True, False]]


def test_word_embedding_numeric_tokens_work_under_mixed_precision():
    embedding = models.WordEmbedding(vocab_size=4, embed_size=4, pad_index=3)
    encoded_number = -2.5 - models.NUMERIC_TOKEN_OFFSET
    inputwords = torch.tensor([[encoded_number, 3.0]])

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output, mask = embedding(inputwords)
        loss = output[0, 0].sum()

    loss.backward()

    assert output.dtype == embedding.vocab_emb.weight.dtype
    assert embedding.float_proj.weight.grad is not None
    assert mask.tolist() == [[True, False]]


def test_rich_measurement_mode_uses_50_percent_dropout():
    assert training.FLEXIBLE_MEASUREMENT_MODE_DROPOUT_PROBABILITY == 0.5
    rows = [
        [
            "Dataset : leaf",
            "Sample : Leaf",
            "Spectral type : Reflectance",
            "Spectrometer : ASD",
            "Measurement mode : Leaf clip",
        ]
    ] * 20_000
    encoded, metadata = training.build_runtime_conditioning(
        rows,
        np.arange(len(rows)),
    )
    conditioned, _ = training.apply_conditioning_schedule(
        torch.from_numpy(encoded),
        metadata["pad_index"],
        generator=torch.Generator().manual_seed(991),
        modes=torch.full(
            (len(rows),),
            training.CONDITIONING_MODE_NAMES.index("flexible_selected_properties"),
        ),
    )

    retained = conditioned[:, 3, 0] != metadata["pad_index"]
    observed_dropout = 1.0 - retained.float().mean().item()
    assert abs(observed_dropout - 0.50) < 0.01


def test_fixed_measurement_mode_uses_independent_50_percent_dropout():
    assert training.FIXED_MEASUREMENT_MODE_DROPOUT_PROBABILITY == 0.5
    rows = [
        [
            "Dataset : leaf",
            "Sample : Leaf",
            "Spectral type : Reflectance",
            "Spectrometer : ASD",
            "Measurement mode : Leaf clip",
            "Water content : 100 g/m2",
            "Leaf mass per area : 50 g/m2",
        ]
    ] * 20_000
    encoded, metadata = training.build_runtime_conditioning(
        rows,
        np.arange(len(rows)),
    )
    conditioned, _ = training.apply_conditioning_schedule(
        torch.from_numpy(encoded),
        metadata["pad_index"],
        generator=torch.Generator().manual_seed(992),
        fixed_mode=training.CONDITIONING_MODE_NAMES.index(
            "sample_spectral_spectrometer_and_measurement_mode"
        ),
        fixed_mode_spectrometer_dropout=False,
        fixed_mode_measurement_dropout=True,
        measurement_mode_dropout_probability=(
            training.FIXED_MEASUREMENT_MODE_DROPOUT_PROBABILITY
        ),
    )

    availability = conditioned[:, :, 0] != metadata["pad_index"]
    assert availability[:, 0].all()
    assert availability[:, 1].all()
    assert availability[:, 2].all()
    assert not availability[:, 4:].any()
    observed_dropout = 1.0 - availability[:, 3].float().mean().item()
    assert abs(observed_dropout - 0.50) < 0.015


def test_fixed_spectrometer_uses_independent_50_percent_dropout():
    rows = [
        [
            "Dataset : soil",
            "Sample : Soil",
            "Spectral type : Reflectance",
            "Spectrometer : ASD",
        ]
    ] * 20_000
    encoded, metadata = training.build_runtime_conditioning(
        rows,
        np.arange(len(rows)),
    )
    spectrometer_generator = torch.Generator().manual_seed(993)
    measurement_generator = torch.Generator().manual_seed(994)
    measurement_state = measurement_generator.get_state().clone()
    conditioned, _ = training.apply_conditioning_schedule(
        torch.from_numpy(encoded),
        metadata["pad_index"],
        spectrometer_generator=spectrometer_generator,
        measurement_generator=measurement_generator,
        fixed_mode=training.CONDITIONING_MODE_NAMES.index(
            "sample_spectral_and_spectrometer"
        ),
        fixed_mode_spectrometer_dropout=True,
        spectrometer_dropout_probability=0.5,
    )

    availability = conditioned[:, :, 0] != metadata["pad_index"]
    assert availability[:, 0].all()
    assert availability[:, 1].all()
    assert not availability[:, 3:].any()
    observed_dropout = 1.0 - availability[:, 2].float().mean().item()
    assert abs(observed_dropout - 0.50) < 0.015
    torch.testing.assert_close(measurement_generator.get_state(), measurement_state)
