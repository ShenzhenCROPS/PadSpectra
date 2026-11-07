import json

import numpy as np
import torch

import trainpadspecmodel_new as training


def test_property_targets_are_dataset_property_note_and_unit_specific(tmp_path):
    rows = [
        [
            "Dataset : D1",
            "Clay content : 10 %",
            "Extractable phosphorus content : 100 mg/kg | Mehlich3 solution",
            "Extractable phosphorus content : 200 mg/kg | NH4C2O4 solution",
            "pH : 6 | Water",
            "pH : 5 | CaCl2 solution",
        ],
        [
            "Dataset : D1",
            "Clay content : 20 %",
            "Extractable phosphorus content : 300 mg/kg | Mehlich3 solution",
            "pH : 8 | Water",
        ],
        ["Dataset : D2", "Clay content : 30 %", "pH : 7 | Water"],
        ["Dataset : D3", "Clay content : 40 %"],
    ]
    report = {
        "rows": len(rows),
        "split": {"test_start_index": 3},
        "target_counts": {
            "Clay content": 4,
            "Extractable phosphorus content": 3,
            "pH": 4,
        },
        "target_units": {
            "Clay content": ["%"],
            "Extractable phosphorus content": ["mg/kg"],
            "pH": [""],
        },
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report))

    head_indices, targets, mask, metadata = training.load_property_targets(
        rows,
        report_path,
        train_rows=3,
        normalization_indices=np.asarray([0, 2]),
        minimum_samples=1,
    )

    heads = {
        (head["dataset"], head["property"], head["condition"])
        for head in metadata["target_heads"]
    }
    assert ("D1", "pH", "Water") in heads
    assert ("D1", "pH", "CaCl2 solution") in heads
    assert ("D2", "pH", "Water") in heads
    assert (
        "D1",
        "Extractable phosphorus content",
        "Mehlich3 solution",
    ) in heads
    assert (
        "D1",
        "Extractable phosphorus content",
        "NH4C2O4 solution",
    ) in heads
    assert not any(head[0] == "D3" for head in heads)
    assert mask[0].sum() == 5
    assert metadata["property_notes_are_separate"] is True
    assert metadata["target_head_identity"] == [
        "dataset",
        "property",
        "note",
        "unit",
    ]
    assert metadata["ph_conditions_are_separate"] is True
    assert np.isfinite(targets[mask]).all()
    assert head_indices.shape == targets.shape == mask.shape

    dense_values, dense_mask = training.dense_property_targets(
        head_indices,
        targets,
        mask,
        metadata,
        np.asarray([0, 1, 2, 3]),
    )
    assert dense_values.shape == dense_mask.shape == (4, len(heads))
    d1_water = next(
        head["index"]
        for head in metadata["target_heads"]
        if head["dataset"] == "D1"
        and head["property"] == "pH"
        and head["condition"] == "Water"
    )
    assert dense_mask[0, d1_water]
    assert dense_values[0, d1_water] == 6
    assert dense_values[1, d1_water] == 8
    mehlich = next(
        head["index"]
        for head in metadata["target_heads"]
        if head["dataset"] == "D1"
        and head["property"] == "Extractable phosphorus content"
        and head["note"] == "Mehlich3 solution"
    )
    oxalate = next(
        head["index"]
        for head in metadata["target_heads"]
        if head["dataset"] == "D1"
        and head["property"] == "Extractable phosphorus content"
        and head["note"] == "NH4C2O4 solution"
    )
    assert dense_values[0, mehlich] == 100
    assert dense_values[0, oxalate] == 200
    assert dense_values[1, mehlich] == 300
    assert not dense_mask[1, oxalate]
    assert not dense_mask[3].any()


def test_snv_linear_heads_backpropagate_to_spectra():
    spectra = torch.rand(3, 11, requires_grad=True)
    head_indices = torch.tensor([[0, 1], [1, 2], [2, 0]])
    predictor = training.DatasetPropertyLinearPredictor(11, 3)
    with torch.no_grad():
        predictor.weight.copy_(
            torch.linspace(-0.2, 0.2, predictor.weight.numel()).reshape_as(
                predictor.weight
            )
        )

    transformed = training.snv_transform(spectra)
    prediction = predictor(transformed, head_indices)
    loss = training.masked_property_loss(
        prediction,
        torch.zeros_like(prediction),
        torch.ones_like(prediction, dtype=torch.bool),
        head_indices,
        target_count=3,
    )
    loss.backward()

    torch.testing.assert_close(transformed.mean(dim=1), torch.zeros(3), atol=1e-6, rtol=0)
    torch.testing.assert_close(
        transformed.std(dim=1, unbiased=False),
        torch.ones(3),
        atol=1e-6,
        rtol=0,
    )
    assert spectra.grad is not None
    assert spectra.grad.abs().sum() > 0
    assert predictor.weight.grad is not None
    assert predictor.weight.grad.abs().sum() > 0


def test_predictor_starts_at_standardized_target_mean_without_random_noise():
    predictor = training.DatasetPropertyLinearPredictor(2151, 7)
    assert torch.count_nonzero(predictor.weight) == 0
    assert torch.count_nonzero(predictor.bias) == 0


def test_supplied_numerical_context_cannot_supervise_itself():
    target_metadata = {
        "target_heads": [
            {"property": "pH", "condition": "Water", "unit": ""},
            {"property": "pH", "condition": "CaCl2 solution", "unit": ""},
            {"property": "pH", "condition": "Unspecified", "unit": ""},
            {"property": "pH", "condition": "KCl solution", "unit": ""},
            {"property": "Water content", "condition": None, "unit": "%"},
            {"property": "Clay content", "condition": None, "unit": "%"},
        ]
    }
    context_slot_by_head = training.property_context_slots(target_metadata)
    pad_index = 99
    context = torch.full(
        (2, training.max_context_entries, training.max_context_tokens),
        pad_index,
        dtype=torch.float32,
    )
    context[0, training.CONTEXT_SLOT_NAMES.index("ph_in_water")] = 1
    context[0, training.CONTEXT_SLOT_NAMES.index("ph_in_cacl2")] = 1
    context[0, training.CONTEXT_SLOT_NAMES.index("ph_generic")] = 1
    context[0, training.CONTEXT_SLOT_NAMES.index("soil_gravimetric_water")] = 1
    head_indices = torch.tensor(
        [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]]
    )
    original_mask = torch.ones_like(head_indices, dtype=torch.bool)

    filtered_mask = training.remove_conditioned_property_targets(
        original_mask,
        head_indices,
        context,
        context_slot_by_head,
        pad_index,
    )

    assert filtered_mask.tolist() == [
        [False, False, False, False, False, True],
        [True, True, True, True, True, True],
    ]
    assert context_slot_by_head.tolist() == [4, 5, 6, 6, 9, -1]


def test_default_predictor_heads_trust_preprocessed_variant_threshold(tmp_path):
    rows = []
    for index in range(40):
        row = [
            "Dataset : D1",
            f"Clay content : {index + 1} %",
            f"pH : {6 + index / 100} | Water",
        ]
        if index < 39:
            row.append(f"pH : {5 + index / 100} | CaCl2 solution")
        rows.append(row)
    rows.extend(
        [
            ["Dataset : D2", f"Clay content : {index + 1} %"]
            for index in range(39)
        ]
    )
    report = {
        "rows": len(rows),
        "split": {"test_start_index": len(rows)},
        "target_counts": {"Clay content": 79, "pH": 79},
        "target_units": {"Clay content": ["%"], "pH": [""]},
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report))

    _, _, _, metadata = training.load_property_targets(
        rows,
        report_path,
        train_rows=len(rows),
        normalization_indices=np.arange(len(rows)),
    )

    included = {head["name"] for head in metadata["target_heads"]}
    assert included == {
        "D1 | Clay content",
        "D1 | pH | Water",
        "D1 | pH | CaCl2 solution",
        "D2 | Clay content",
    }
    assert metadata["minimum_samples_per_head"] == 1
    assert metadata["head_selection_policy"] == "at_least_one_training_label"


def test_property_manifest_is_verified_once_before_training(tmp_path):
    rows = [
        ["Dataset : D1", "Sample : Soil", f"Clay content : {index} %"]
        for index in range(40)
    ]
    manifest_path = tmp_path / "processing_manifest.json"
    schema_path = tmp_path / "target_schema.json"
    manifest_path.write_text(
        json.dumps(
            {
                "profile": "unit-test",
                "minimum_property_variant_count": 40,
            }
        )
    )
    schema_path.write_text(
        json.dumps(
            {
                "target_properties": [
                    {"sample": "Soil", "label": "Clay content", "unit": "%"}
                ]
            }
        )
    )

    audit = training.verify_dataset_property_manifest(
        rows,
        manifest_path,
        schema_path,
    )
    assert audit["retained_property_variant_count"] == 1
    assert audit["minimum_property_variant_count"] == 40

    try:
        training.verify_dataset_property_manifest(
            rows[:-1],
            manifest_path,
            schema_path,
        )
    except ValueError as error:
        assert "below 40" in str(error)
    else:
        raise AssertionError("Undersized retained property variant was accepted")


def test_wet_soil_uses_only_gravimetric_water_for_training_constraint(tmp_path):
    rows = []
    for index in range(45):
        rows.append(
            [
                "Dataset : Wet",
                "Sample : Soil",
                f"Water content : {index + 1} %",
                f"Clay content : {10 + index} %",
                f"pH : {6 + index / 100} | Water",
            ]
        )
        rows.append(
            [
                "Dataset : Dry",
                "Sample : Soil",
                "Water content : 0 %",
                f"Clay content : {10 + index} %",
                f"pH : {6 + index / 100} | Water",
            ]
        )
    report = {
        "rows": len(rows),
        "split": {"test_start_index": len(rows)},
        "target_counts": {
            "Water content": len(rows),
            "Clay content": len(rows),
            "pH": len(rows),
        },
        "target_units": {
            "Water content": ["%"],
            "Clay content": ["%"],
            "pH": [""],
        },
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report))

    _, _, _, metadata = training.load_property_targets(
        rows,
        report_path,
        train_rows=len(rows),
        normalization_indices=np.arange(len(rows)),
        minimum_samples=40,
    )

    included = {head["name"] for head in metadata["target_heads"]}
    assert "Wet | Water content" in included
    assert "Wet | Clay content" not in included
    assert "Wet | pH | Water" not in included
    assert "Dry | Clay content" in included
    assert "Dry | pH | Water" in included
    wet_excluded = {
        head["name"]: head["reason"]
        for head in metadata["excluded_target_heads"]
        if head["dataset"] == "Wet"
    }
    assert wet_excluded == {
        "Wet | Clay content": "wet_soil_non_moisture_property",
        "Wet | pH | Water": "wet_soil_non_moisture_property",
    }
    assert metadata["wet_soil_constraint"]["datasets"] == ["Wet"]
