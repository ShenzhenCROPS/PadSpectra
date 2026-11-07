import torch
import json
import torch.nn as nn
import torch.nn.functional as F
import h5py
import numpy as np
import torch.optim as optim
import torch.nn.init as init
import padspecmodels_new as psm
from torch.utils.data import DataLoader, Dataset, Subset
from torch.optim.lr_scheduler import LambdaLR
import math
import copy
import hashlib
import os
import re
from collections import Counter
from pathlib import Path
from torch.amp import GradScaler, autocast

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR_ENV = "PADSPECTRA_DATA_DIR"
DATA_DIR = Path(os.environ.get(DATA_DIR_ENV, ROOT_DIR / "Data")).resolve()
TRAINING_VARIANT_ENV = "PADSPECTRA_TRAINING_VARIANT"
RUN_NAME_ENV = "PADSPECTRA_RUN_NAME"
RUNS_DIR_ENV = "PADSPECTRA_RUNS_DIR"
CONTEXT_ENCODING_ENV = "PADSPECTRA_CONTEXT_ENCODING"
ATOMIC_CONTEXT_ENCODING = "atomic_v1"
WORD_SPLIT_CONTEXT_ENCODING = "word_split_v2"
CONTEXT_ENCODING_NAMES = (
    ATOMIC_CONTEXT_ENCODING,
    WORD_SPLIT_CONTEXT_ENCODING,
)
CONTEXT_ENCODING = os.environ.get(
    CONTEXT_ENCODING_ENV,
    WORD_SPLIT_CONTEXT_ENCODING,
)
if CONTEXT_ENCODING not in CONTEXT_ENCODING_NAMES:
    raise ValueError(
        f"Unknown {CONTEXT_ENCODING_ENV}={CONTEXT_ENCODING!r}; expected one of "
        f"{', '.join(CONTEXT_ENCODING_NAMES)}"
    )
TRAINING_VARIANTS = {
    "flexible_pure_dense_ffn": {
        "conditioning_mode": "flexible_selected_properties",
        "property_constraint": False,
        "spectrometer_dropout": True,
        "measurement_mode_dropout": True,
    },
}
TRAINING_VARIANT_NAMES = tuple(TRAINING_VARIANTS)
TRAINING_VARIANT = os.environ.get(
    TRAINING_VARIANT_ENV,
    "flexible_pure_dense_ffn",
)
if TRAINING_VARIANT not in TRAINING_VARIANTS:
    raise ValueError(
        f"Unknown {TRAINING_VARIANT_ENV}={TRAINING_VARIANT!r}; expected one of "
        f"{', '.join(TRAINING_VARIANT_NAMES)}"
    )
TRAINING_VARIANT_CONFIG = TRAINING_VARIANTS[TRAINING_VARIANT]
TRAINING_RUN_NAME = os.environ.get(RUN_NAME_ENV, TRAINING_VARIANT)
if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", TRAINING_RUN_NAME) is None:
    raise ValueError(
        f"Invalid {RUN_NAME_ENV}={TRAINING_RUN_NAME!r}; use a single safe "
        "directory name containing letters, numbers, dots, dashes, or underscores"
    )
RUNS_DIR = Path(os.environ.get(RUNS_DIR_ENV, ROOT_DIR / "runs")).resolve()
RUN_DIR = RUNS_DIR / TRAINING_RUN_NAME
LOSS_DIR = RUN_DIR / "loss"
WEIGHTS_DIR = RUN_DIR / "weights"
SPECTRA_PATH = DATA_DIR / "spectra.h5"
METADATA_PATH = DATA_DIR / "metadata.json"
SPLIT_INDEX_PATH = DATA_DIR / "split_indices.json"
PROCESSING_MANIFEST_PATH = DATA_DIR / "processing_manifest.json"
TARGET_SCHEMA_PATH = DATA_DIR / "target_schema.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
amp_enabled = device.type == "cuda"
if amp_enabled:
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

bandnum = 43
bandwidth = 50
minimum_observed_bands = 8
maximum_missing_bands = bandnum - minimum_observed_bands
max_context_entries = 10
CONTEXT_TOKEN_COUNTS = {
    ATOMIC_CONTEXT_ENCODING: 2,
    WORD_SPLIT_CONTEXT_ENCODING: 6,
}
max_context_tokens = CONTEXT_TOKEN_COUNTS[CONTEXT_ENCODING]

hidden_dim = 180
property_start_epoch = 1000
property_loss_weight = 0.0005
property_input_spectra = "snv_raw_padspectra_output"
property_predictor_type = "dataset_property_linear"
property_snv_epsilon = 1e-6
minimum_property_samples = 40
wet_soil_moisture_target = "Water content"
training_checkpoint_format_version = (
    21 if CONTEXT_ENCODING == WORD_SPLIT_CONTEXT_ENCODING else 20
)

ema_decay = 0.99
ema_update_interval = 5

batchsize = 3500
learningRate = 0.0003
TRAINING_END_EPOCH_ENV = "PADSPECTRA_NUM_EPOCHS"
LR_PLATEAU_END_EPOCH_ENV = "PADSPECTRA_LR_PLATEAU_END_EPOCH"
INITIAL_SCHEDULE_END_EPOCH_ENV = "PADSPECTRA_INITIAL_SCHEDULE_END_EPOCH"
num_epochs = int(os.environ.get(TRAINING_END_EPOCH_ENV, "25000"))
lr_plateau_end_epoch = int(os.environ.get(LR_PLATEAU_END_EPOCH_ENV, "-1"))
initial_schedule_end_epoch = int(
    os.environ.get(INITIAL_SCHEDULE_END_EPOCH_ENV, "15000")
)
if num_epochs < 1:
    raise ValueError(f"{TRAINING_END_EPOCH_ENV} must be positive")
if not 1 <= initial_schedule_end_epoch <= num_epochs:
    raise ValueError(
        f"{INITIAL_SCHEDULE_END_EPOCH_ENV} must be between 1 and "
        f"{TRAINING_END_EPOCH_ENV} ({num_epochs}), inclusive"
    )
if not -1 <= lr_plateau_end_epoch < initial_schedule_end_epoch:
    raise ValueError(
        f"{LR_PLATEAU_END_EPOCH_ENV} must be -1 (disabled) or between 0 "
        f"and {INITIAL_SCHEDULE_END_EPOCH_ENV} - 1 "
        f"({initial_schedule_end_epoch - 1}), inclusive"
    )
continuation_final_learning_rate = 0.00001
max_grad_norm = 1.0

warmup_steps = 500
validation_interval = 10
validation_mask_seed = 314159
validation_mask_epoch = 1000
validation_conditioning_policy = "all_available_context"
best_validation_metric = "masked_reconstruction_mae_v2"
secondary_validation_metric = "masked_reconstruction_mae_plus_property_v2"
checkpoint_interval = 100
ema_checkpoint_interval = 1000
metrics_flush_interval = validation_interval
resume_checkpoint_env = "PADSPECTRA_RESUME_CHECKPOINT"


def learning_rate_factor(current_step, steps_per_epoch):
    """Warm up, optionally hold, then run one or two cosine decays.

    After the optimizer-step warm-up, the base learning rate is held through
    ``lr_plateau_end_epoch`` when that value is non-negative. The primary cosine
    then reaches 10% of the base learning rate at the configured initial
    schedule end. If training continues beyond that epoch, a second cosine
    segment starts continuously and reaches the continuation final rate.
    """
    if steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch must be positive")
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))

    plateau_total_steps = warmup_steps
    if lr_plateau_end_epoch >= 0:
        plateau_total_steps = max(
            warmup_steps,
            (lr_plateau_end_epoch + 1) * steps_per_epoch,
        )
    if current_step <= plateau_total_steps:
        return 1.0

    initial_total_steps = (initial_schedule_end_epoch + 1) * steps_per_epoch
    initial_min_factor = 0.1
    if current_step <= initial_total_steps:
        progress = float(current_step - plateau_total_steps) / float(
            max(1, initial_total_steps - plateau_total_steps)
        )
        return initial_min_factor + (1 - initial_min_factor) * 0.5 * (
            1 + math.cos(math.pi * progress)
        )

    continuation_steps = max(
        1,
        (num_epochs - initial_schedule_end_epoch) * steps_per_epoch,
    )
    continuation_progress = min(
        1.0,
        float(current_step - initial_total_steps) / float(continuation_steps),
    )
    final_factor = continuation_final_learning_rate / learningRate
    return final_factor + (initial_min_factor - final_factor) * 0.5 * (
        1 + math.cos(math.pi * continuation_progress)
    )

TARGET_VALUE_PATTERN = re.compile(
    r"^\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*(.*?)\s*$"
)
PH_CONDITION_DISPLAY_ORDER = (
    "Water",
    "CaCl2 solution",
    "KCl solution",
    "Acetate",
    "Unspecified",
)

CONTEXT_SLOT_NAMES = (
    "sample_type",
    "spectral_type",
    "spectrometer",
    "measurement_mode",
    "ph_in_water",
    "ph_in_cacl2",
    "ph_generic",
    "leaf_water",
    "leaf_lma",
    "soil_gravimetric_water",
)
CONDITIONING_MODE_NAMES = (
    "spectra_only",
    "sample_type",
    "sample_and_spectral_type",
    "sample_spectral_and_spectrometer",
    "sample_spectral_spectrometer_and_measurement_mode",
    "flexible_selected_properties",
)
TRAINING_CONDITIONING_MODE = TRAINING_VARIANT_CONFIG["conditioning_mode"]
TRAINING_CONDITIONING_MODE_INDEX = CONDITIONING_MODE_NAMES.index(
    TRAINING_CONDITIONING_MODE
)
PROPERTY_CONSTRAINT_ENABLED = TRAINING_VARIANT_CONFIG["property_constraint"]
SPECTROMETER_DROPOUT_ENABLED = TRAINING_VARIANT_CONFIG[
    "spectrometer_dropout"
]
MEASUREMENT_MODE_DROPOUT_ENABLED = TRAINING_VARIANT_CONFIG[
    "measurement_mode_dropout"
]
TRAINING_FEED_FORWARD_TYPE = psm.FEED_FORWARD_DENSE
PERIODIC_EMA_SNAPSHOTS_ENABLED = True
# Keep initialization, data order, and stochastic masking aligned across the
# independent conditioning variants so their validation results are directly
# comparable. The models still train independently and write separate state.
TRAINING_SEED = 20260804
CONTEXT_FIELD_TOKENS = {
    "sample_type": "field:sample_type",
    "spectral_type": "field:spectral_type",
    "spectrometer": "field:spectrometer",
    "measurement_mode": "field:measurement_mode",
    "ph_in_water": "field:ph_in_water",
    "ph_in_cacl2": "field:ph_in_cacl2",
    "ph_generic": "field:ph_generic",
    "leaf_water": "field:leaf_water",
    "leaf_lma": "field:leaf_lma",
    "soil_gravimetric_water": "field:soil_gravimetric_water",
}
WORD_SPLIT_CONTEXT_LABELS = {
    "sample_type": ("sample",),
    "spectral_type": ("spectral", "type"),
    "spectrometer": ("spectrometer",),
    "measurement_mode": ("measurement", "mode"),
    "ph_in_water": ("ph",),
    "ph_in_cacl2": ("ph",),
    "ph_generic": ("ph",),
    "leaf_water": ("water", "content"),
    "leaf_lma": ("leaf", "mass", "per", "area"),
    "soil_gravimetric_water": ("water", "content"),
}
WORD_SPLIT_CONTEXT_SUFFIXES = {
    "ph_in_water": ("water",),
    "ph_in_cacl2": ("cacl2", "solution"),
    "ph_generic": (),
    "leaf_water": ("g/m2",),
    "leaf_lma": ("g/m2",),
    "soil_gravimetric_water": ("%",),
}
NUMERICAL_CONTEXT_SLOTS = frozenset(CONTEXT_SLOT_NAMES[4:])
_NUMERICAL_CONTEXT_VALUE = object()
SPECTROMETER_DROPOUT_PROBABILITY = 0.50
MEASUREMENT_MODE_DROPOUT_PROBABILITY = 0.50
FLEXIBLE_MEASUREMENT_MODE_DROPOUT_PROBABILITY = (
    MEASUREMENT_MODE_DROPOUT_PROBABILITY
)
FIXED_MEASUREMENT_MODE_DROPOUT_PROBABILITY = (
    MEASUREMENT_MODE_DROPOUT_PROBABILITY
)
RICH_NUMERICAL_PROPERTY_RETENTION_PROBABILITY = 0.5
TRAINING_SPECTROMETER_DROPOUT_PROBABILITY = (
    SPECTROMETER_DROPOUT_PROBABILITY if SPECTROMETER_DROPOUT_ENABLED else 0.0
)
TRAINING_MEASUREMENT_MODE_DROPOUT_PROBABILITY = (
    MEASUREMENT_MODE_DROPOUT_PROBABILITY
    if MEASUREMENT_MODE_DROPOUT_ENABLED
    else 0.0
)

TRAINING_GENERATOR_SEEDS = {
    "data_order": TRAINING_SEED + 101,
    "spectral_mask": TRAINING_SEED + 102,
    "spectrometer_dropout": TRAINING_SEED + 103,
    "measurement_dropout": TRAINING_SEED + 104,
    "property_dropout": TRAINING_SEED + 105,
}


def _split_target_item(item):
    if not isinstance(item, str) or " : " not in item:
        return None
    name, payload = item.split(" : ", 1)
    value_and_unit, separator, condition = payload.partition(" | ")
    match = TARGET_VALUE_PATTERN.match(value_and_unit)
    if match is None:
        return None
    value = float(match.group(1))
    unit = match.group(2).strip()
    condition = condition.strip() if separator else ""
    return name, value, unit, condition


def _canonical_target_note(name, note):
    """Preserve every measurement note as part of a target's identity."""
    if note:
        return note
    # Retain the established explicit label for note-free pH heads. Other
    # note-free properties keep an empty key and an absent note in metadata.
    return "Unspecified" if name == "pH" else ""


def _dataset_label(metadata_row):
    labels = [
        item.split(" : ", 1)[1]
        for item in metadata_row
        if isinstance(item, str) and item.startswith("Dataset : ")
    ]
    if len(labels) != 1:
        raise ValueError(
            f"Expected exactly one dataset label in metadata row, found {labels}"
        )
    return labels[0]


def _single_metadata_value(metadata_row, label):
    values = []
    prefix = f"{label} : "
    for item in metadata_row:
        if isinstance(item, str) and item.startswith(prefix):
            values.append(item[len(prefix) :])
    unique = list(dict.fromkeys(values))
    if len(unique) > 1:
        raise ValueError(f"Conflicting {label} values in metadata row: {unique}")
    return unique[0] if unique else None


def _normalized_category_token(label, value):
    normalized_value = " ".join(value.split()).casefold()
    return f"{label}:{normalized_value}"


def _split_context_words(value):
    """Normalize one categorical value into reusable word tokens."""
    return tuple(token.casefold() for token in " ".join(value.split()).split())


def _word_split_context_entry(slot_name, value):
    """Return semantic word tokens plus a scalar marker for one entry.

    Separators such as ``:`` and ``|`` are intentionally omitted: the entry
    boundary already supplies that structure. Property labels, units, and the
    two retained pH conditions remain explicit tokens, while the numerical
    value is represented by the same standardized scalar projection used by
    the original encoding.
    """
    tokens = list(WORD_SPLIT_CONTEXT_LABELS[slot_name])
    if slot_name in NUMERICAL_CONTEXT_SLOTS:
        tokens.append(_NUMERICAL_CONTEXT_VALUE)
        tokens.extend(WORD_SPLIT_CONTEXT_SUFFIXES[slot_name])
    else:
        tokens.extend(_split_context_words(value))
    return tuple(tokens)


def _selected_condition_values(metadata_row):
    """Return the ten fixed conditioning slots for one raw metadata row.

    Water and CaCl2 pH retain dedicated numerical slots. Note-free pH and all
    other pH methods share the generic pH input slot after their notes are
    stripped for conditioning only. Original notes remain untouched in the
    predictor targets. Repeated values contributing to one slot are averaged.
    """
    sample = _single_metadata_value(metadata_row, "Sample")
    spectral_type = _single_metadata_value(metadata_row, "Spectral type")
    spectrometer = _single_metadata_value(metadata_row, "Spectrometer")
    measurement_mode = _single_metadata_value(metadata_row, "Measurement mode")
    numerical_values = {
        "ph_in_water": [],
        "ph_in_cacl2": [],
        "ph_generic": [],
        "leaf_water": [],
        "leaf_lma": [],
        "soil_gravimetric_water": [],
    }
    for item in metadata_row:
        parsed = _split_target_item(item)
        if parsed is None:
            continue
        name, value, _unit, condition = parsed
        if sample == "Soil" and name == "pH" and condition == "Water":
            numerical_values["ph_in_water"].append(value)
        elif (
            sample == "Soil"
            and name == "pH"
            and condition == "CaCl2 solution"
        ):
            numerical_values["ph_in_cacl2"].append(value)
        elif sample == "Soil" and name == "pH":
            numerical_values["ph_generic"].append(value)
        elif sample == "Leaf" and name == "Water content" and _unit == "g/m2":
            numerical_values["leaf_water"].append(value)
        elif sample == "Leaf" and name == "Leaf mass per area":
            numerical_values["leaf_lma"].append(value)
        elif (
            sample == "Soil"
            and name == wet_soil_moisture_target
            and _unit == "%"
        ):
            numerical_values["soil_gravimetric_water"].append(value)

    selected = {
        "sample_type": sample,
        "spectral_type": spectral_type,
        "spectrometer": spectrometer,
        "measurement_mode": measurement_mode,
    }
    for name, values in numerical_values.items():
        selected[name] = float(np.mean(values)) if values else None
    return selected


def build_runtime_conditioning(
    metadata_rows,
    normalization_indices,
    context_encoding=None,
):
    """Parse raw metadata and encode model context inside the trainer.

    Categorical vocabulary items and numerical normalization statistics are
    learned only from optimization rows. There is no unknown token: missing or
    validation-only categorical values become padding. ``atomic_v1`` preserves
    existing checkpoints. ``word_split_v2`` represents labels, multiword
    categorical values, units, and retained pH notes as reusable word tokens.
    """
    if context_encoding is None:
        context_encoding = CONTEXT_ENCODING
    if context_encoding not in CONTEXT_ENCODING_NAMES:
        raise ValueError(
            f"Unknown context encoding {context_encoding!r}; expected one of "
            f"{', '.join(CONTEXT_ENCODING_NAMES)}"
        )
    context_token_count = CONTEXT_TOKEN_COUNTS[context_encoding]
    normalization_indices = np.asarray(normalization_indices, dtype=np.int64)
    if (
        normalization_indices.ndim != 1
        or normalization_indices.size == 0
        or normalization_indices.min() < 0
        or normalization_indices.max() >= len(metadata_rows)
        or np.unique(normalization_indices).size != normalization_indices.size
    ):
        raise ValueError("Conditioning normalization indices are invalid")

    parsed_rows = [_selected_condition_values(row) for row in metadata_rows]
    optimization_rows = [parsed_rows[index] for index in normalization_indices]

    if context_encoding == ATOMIC_CONTEXT_ENCODING:
        vocabulary_tokens = set(CONTEXT_FIELD_TOKENS.values())
        for values in optimization_rows:
            for slot_name in CONTEXT_SLOT_NAMES[:4]:
                value = values[slot_name]
                if value is not None:
                    vocabulary_tokens.add(
                        _normalized_category_token(slot_name, value)
                    )
    else:
        vocabulary_tokens = set()
        for slot_name in CONTEXT_SLOT_NAMES:
            placeholder = 0.0 if slot_name in NUMERICAL_CONTEXT_SLOTS else ""
            vocabulary_tokens.update(
                token
                for token in _word_split_context_entry(slot_name, placeholder)
                if token is not _NUMERICAL_CONTEXT_VALUE
            )
        for values in optimization_rows:
            for slot_name in CONTEXT_SLOT_NAMES[:4]:
                value = values[slot_name]
                if value is not None:
                    vocabulary_tokens.update(
                        _word_split_context_entry(slot_name, value)
                    )
    word2idx = {
        token: index for index, token in enumerate(sorted(vocabulary_tokens))
    }
    pad_index = len(word2idx)

    numerical_statistics = {}
    for slot_name in CONTEXT_SLOT_NAMES[4:]:
        values = np.asarray(
            [
                row[slot_name]
                for row in optimization_rows
                if row[slot_name] is not None
            ],
            dtype=np.float64,
        )
        if values.size:
            mean = float(values.mean())
            std = float(values.std())
            if std <= 1e-8:
                std = 1.0
        else:
            mean = 0.0
            std = 1.0
        numerical_statistics[slot_name] = {
            "mean": mean,
            "std": std,
            "count": int(values.size),
        }

    encoded, unseen_categories = _encode_conditioning_rows(
        parsed_rows, context_encoding, word2idx, pad_index, numerical_statistics
    )

    metadata = {
        "slot_names": list(CONTEXT_SLOT_NAMES),
        "mode_names": list(CONDITIONING_MODE_NAMES),
        "spectrometer_dropout_probability": SPECTROMETER_DROPOUT_PROBABILITY,
        "flexible_measurement_mode_dropout_probability": (
            FLEXIBLE_MEASUREMENT_MODE_DROPOUT_PROBABILITY
        ),
        "fixed_measurement_mode_dropout_probability": (
            FIXED_MEASUREMENT_MODE_DROPOUT_PROBABILITY
        ),
        "rich_numerical_property_retention_probability": (
            RICH_NUMERICAL_PROPERTY_RETENTION_PROBABILITY
        ),
        "vocabulary_fitted_on": "optimization_rows_only",
        "word2idx": word2idx,
        "idx2word": {str(index): token for token, index in word2idx.items()},
        "pad_index": pad_index,
        "vocab_size": pad_index + 1,
        "unknown_token_defined": False,
        "unseen_categories_padded": sorted(unseen_categories),
        "numerical_statistics": numerical_statistics,
        "numeric_token_offset": psm.NUMERIC_TOKEN_OFFSET,
        "numerical_properties": {
            "ph_in_water": {
                "sample": "Soil",
                "label": "pH",
                "note": "Water",
                "unit": "",
            },
            "ph_in_cacl2": {
                "sample": "Soil",
                "label": "pH",
                "note": "CaCl2 solution",
                "unit": "",
            },
            "ph_generic": {
                "sample": "Soil",
                "label": "pH",
                "note": None,
                "unit": "",
                "conditioning_note_policy": (
                    "no note plus all notes other than Water and CaCl2 solution"
                ),
            },
            "leaf_water": {
                "sample": "Leaf",
                "label": "Water content",
                "unit": "g/m2",
            },
            "leaf_lma": {
                "sample": "Leaf",
                "label": "Leaf mass per area",
                "unit": "g/m2",
            },
            "soil_gravimetric_water": {
                "sample": "Soil",
                "label": wet_soil_moisture_target,
                "unit": "%",
            },
        },
    }
    if context_encoding == WORD_SPLIT_CONTEXT_ENCODING:
        metadata.update(
            {
                "context_encoding": WORD_SPLIT_CONTEXT_ENCODING,
                "max_context_tokens": context_token_count,
                "tokenization": "normalized_whitespace_words_without_separators",
                "numerical_value_encoding": "training_zscore_float_projection",
                "unseen_context_words_padded": sorted(unseen_categories),
            }
        )
    return encoded, metadata


def encode_runtime_conditioning(metadata_rows, context_metadata):
    """Encode new metadata with the vocabulary and statistics from training."""
    parsed_rows = [_selected_condition_values(row) for row in metadata_rows]
    context_encoding = context_metadata.get(
        "context_encoding", ATOMIC_CONTEXT_ENCODING
    )
    encoded, _ = _encode_conditioning_rows(
        parsed_rows,
        context_encoding,
        context_metadata["word2idx"],
        context_metadata["pad_index"],
        context_metadata["numerical_statistics"],
    )
    return encoded


def _encode_conditioning_rows(
    parsed_rows, context_encoding, word2idx, pad_index, numerical_statistics
):
    context_token_count = CONTEXT_TOKEN_COUNTS[context_encoding]
    encoded = np.full(
        (len(parsed_rows), max_context_entries, context_token_count),
        pad_index,
        dtype=np.float32,
    )
    unseen_categories = set()
    for row_index, values in enumerate(parsed_rows):
        for slot_index, slot_name in enumerate(CONTEXT_SLOT_NAMES):
            value = values[slot_name]
            if value is None:
                continue
            if context_encoding == ATOMIC_CONTEXT_ENCODING:
                encoded[row_index, slot_index, 0] = word2idx[
                    CONTEXT_FIELD_TOKENS[slot_name]
                ]
                if slot_index < 4:
                    token = _normalized_category_token(slot_name, value)
                    value_index = word2idx.get(token)
                    if value_index is None:
                        encoded[row_index, slot_index] = pad_index
                        unseen_categories.add(token)
                    else:
                        encoded[row_index, slot_index, 1] = value_index
                    continue
                statistics = numerical_statistics[slot_name]
                normalized_value = (value - statistics["mean"]) / statistics["std"]
                if abs(normalized_value) >= psm.NUMERIC_TOKEN_OFFSET / 2:
                    raise ValueError(
                        f"Normalized {slot_name} value is outside the supported "
                        f"numeric token range: {normalized_value}"
                    )
                encoded[row_index, slot_index, 1] = (
                    -normalized_value - psm.NUMERIC_TOKEN_OFFSET
                )
                continue

            entry_tokens = _word_split_context_entry(slot_name, value)
            if len(entry_tokens) > context_token_count:
                raise ValueError(
                    f"Word-split context entry {slot_name!r} requires "
                    f"{len(entry_tokens)} tokens, exceeding {context_token_count}"
                )
            for token_index, token in enumerate(entry_tokens):
                if token is _NUMERICAL_CONTEXT_VALUE:
                    statistics = numerical_statistics[slot_name]
                    normalized_value = (
                        value - statistics["mean"]
                    ) / statistics["std"]
                    if abs(normalized_value) >= psm.NUMERIC_TOKEN_OFFSET / 2:
                        raise ValueError(
                            f"Normalized {slot_name} value is outside the "
                            f"supported numeric token range: {normalized_value}"
                        )
                    encoded[row_index, slot_index, token_index] = (
                        -normalized_value - psm.NUMERIC_TOKEN_OFFSET
                    )
                    continue
                value_index = word2idx.get(token)
                if value_index is None:
                    unseen_categories.add(token)
                else:
                    encoded[row_index, slot_index, token_index] = value_index

    return encoded, unseen_categories


def apply_conditioning_schedule(
    encoded_context,
    pad_index,
    generator=None,
    spectrometer_generator=None,
    measurement_generator=None,
    property_generator=None,
    modes=None,
    fixed_mode=None,
    fixed_mode_spectrometer_dropout=False,
    fixed_mode_measurement_dropout=False,
    mask_optional_context=True,
    spectrometer_dropout_probability=SPECTROMETER_DROPOUT_PROBABILITY,
    measurement_mode_dropout_probability=(
        FLEXIBLE_MEASUREMENT_MODE_DROPOUT_PROBABILITY
    ),
    numerical_property_retention_probability=(
        RICH_NUMERICAL_PROPERTY_RETENTION_PROBABILITY
    ),
):
    """Apply one conditioning regime to a batch.

    Training supplies ``fixed_mode`` so an entire run has one invariant
    conditioning design. ``modes`` remains available for focused tests and
    offline analysis, but the trainer never mixes model variants by row.
    """
    if (
        encoded_context.ndim != 3
        or encoded_context.shape[1] != max_context_entries
        or encoded_context.shape[2] not in set(CONTEXT_TOKEN_COUNTS.values())
    ):
        raise ValueError(
            "Encoded context must have shape (batch, "
            f"{max_context_entries}, one of "
            f"{sorted(set(CONTEXT_TOKEN_COUNTS.values()))})"
        )
    if not 0 <= spectrometer_dropout_probability <= 1:
        raise ValueError("Spectrometer dropout probability must be in [0, 1]")
    if not 0 <= measurement_mode_dropout_probability <= 1:
        raise ValueError("Measurement-mode dropout probability must be in [0, 1]")
    if not 0 <= numerical_property_retention_probability <= 1:
        raise ValueError(
            "Numerical-property retention probability must be in [0, 1]"
        )

    batch_rows = encoded_context.shape[0]
    if modes is not None and fixed_mode is not None:
        raise ValueError("Use either per-row modes or one fixed mode, not both")
    if modes is None:
        if fixed_mode is None:
            raise ValueError("A fixed conditioning mode is required")
        modes = torch.full((batch_rows,), int(fixed_mode), dtype=torch.long)
    else:
        modes = torch.as_tensor(modes, dtype=torch.long).cpu()
        if modes.shape != (batch_rows,):
            raise ValueError("One conditioning mode is required per batch row")
    if (modes < 0).any() or (modes >= len(CONDITIONING_MODE_NAMES)).any():
        raise ValueError("Conditioning mode index is out of range")

    spectrometer_generator = spectrometer_generator or generator
    measurement_generator = measurement_generator or generator
    property_generator = property_generator or generator

    selected = torch.zeros(
        batch_rows,
        max_context_entries,
        dtype=torch.bool,
    )
    selected[modes >= 1, 0] = True
    selected[modes >= 2, 1] = True
    selected[modes >= 3, 2] = True
    selected[modes >= 4, 3] = True

    spectrometer_rows = modes >= 3
    spectrometer_dropout_rows = spectrometer_rows & (
        (modes == len(CONDITIONING_MODE_NAMES) - 1)
        | bool(fixed_mode_spectrometer_dropout)
    )
    if mask_optional_context and spectrometer_dropout_rows.any():
        spectrometer_draws = torch.rand(
            int(spectrometer_dropout_rows.sum()),
            generator=spectrometer_generator,
        )
        selected[spectrometer_dropout_rows, 2] = (
            spectrometer_draws >= spectrometer_dropout_probability
        )

    measurement_rows = modes >= 4
    measurement_dropout_rows = measurement_rows & (
        (modes == len(CONDITIONING_MODE_NAMES) - 1)
        | bool(fixed_mode_measurement_dropout)
    )
    if mask_optional_context and measurement_dropout_rows.any():
        measurement_draws = torch.rand(
            int(measurement_dropout_rows.sum()),
            generator=measurement_generator,
        )
        selected[measurement_dropout_rows, 3] = (
            measurement_draws >= measurement_mode_dropout_probability
        )

    rich_rows = modes == len(CONDITIONING_MODE_NAMES) - 1
    if rich_rows.any():
        if mask_optional_context:
            property_draws = torch.rand(
                int(rich_rows.sum()),
                max_context_entries - 4,
                generator=property_generator,
            )
            selected[rich_rows, 4:] = (
                property_draws < numerical_property_retention_probability
            )
        else:
            selected[rich_rows, 4:] = True

    selected = selected.to(encoded_context.device)
    availability = encoded_context[:, :, 0] != pad_index
    selected &= availability
    selected = selected.unsqueeze(-1)
    conditioned = torch.where(
        selected,
        encoded_context,
        torch.full_like(encoded_context, pad_index),
    )
    return conditioned, modes


def load_predefined_split(
    metadata_rows,
    index_path,
    row_count,
    fraction=None,
    seed=None,
):
    """Load and audit the predefined indexed train/validation/test split.

    ``fraction`` and ``seed`` are accepted only for compatibility with older
    evaluation callers. The trainer never resamples the supplied partitions.
    """
    del fraction, seed
    with Path(index_path).open() as file:
        split = json.load(file)
    if split.get("row_count") != row_count or len(metadata_rows) != row_count:
        raise ValueError(
            "split_indices.json, metadata.json, and spectra.h5 row counts must "
            f"match; split={split.get('row_count')}, metadata={len(metadata_rows)}, "
            f"configured={row_count}"
        )

    partitions = {}
    for name in ("train", "validation", "test"):
        try:
            definition = split["partitions"][name]
            indices = np.asarray(definition["indices"], dtype=np.int64)
        except KeyError as error:
            raise ValueError(f"Indexed split is missing partition {name!r}") from error
        if indices.ndim != 1 or indices.size != definition["rows"]:
            raise ValueError(f"Invalid index count for partition {name!r}")
        if indices.size == 0 or indices.min() < 0 or indices.max() >= row_count:
            raise ValueError(f"Out-of-range or empty partition {name!r}")
        if np.unique(indices).size != indices.size:
            raise ValueError(f"Duplicate row indices in partition {name!r}")

        observed_counts = {}
        for row_index in indices:
            dataset = _dataset_label(metadata_rows[int(row_index)])
            observed_counts[dataset] = observed_counts.get(dataset, 0) + 1
        if observed_counts != definition["dataset_row_counts"]:
            raise ValueError(
                f"Metadata dataset counts do not match partition {name!r}"
            )
        partitions[name] = indices

    membership = np.concatenate(tuple(partitions.values()))
    if membership.size != row_count or np.unique(membership).size != row_count:
        raise ValueError(
            "Train, validation, and test indices must be exhaustive and disjoint"
        )
    if not np.array_equal(np.sort(membership), np.arange(row_count)):
        raise ValueError("Indexed partitions do not cover every aligned row exactly once")

    index_bytes = Path(index_path).read_bytes()
    train_definition = split["partitions"]["train"]
    validation_definition = split["partitions"]["validation"]
    test_definition = split["partitions"]["test"]
    split_metadata = {
        "strategy": "predefined_index_file",
        "processing_profile": split.get("processing_profile"),
        "index_semantics": split.get("index_semantics"),
        "row_count": row_count,
        "training_rows": int(partitions["train"].size),
        "validation_rows": int(partitions["validation"].size),
        "test_rows": int(partitions["test"].size),
        "source_dataset_count": len(train_definition["dataset_labels"]),
        "validation_source_dataset_count": len(
            validation_definition["dataset_labels"]
        ),
        "test_source_dataset_count": len(test_definition["dataset_labels"]),
        "source_training_counts": train_definition["dataset_row_counts"],
        "source_validation_counts": validation_definition["dataset_row_counts"],
        "source_test_counts": test_definition["dataset_row_counts"],
        "split_index_sha256": hashlib.sha256(index_bytes).hexdigest(),
        "training_index_sha256": hashlib.sha256(
            partitions["train"].tobytes()
        ).hexdigest(),
        "validation_index_sha256": hashlib.sha256(
            partitions["validation"].tobytes()
        ).hexdigest(),
        "test_index_sha256": hashlib.sha256(
            partitions["test"].tobytes()
        ).hexdigest(),
    }
    return partitions["train"], partitions["validation"], split_metadata


def build_source_stratified_split(
    metadata_rows,
    index_path,
    row_count,
    fraction=None,
    seed=None,
):
    """Compatibility wrapper for callers written for the former split API."""
    return load_predefined_split(
        metadata_rows,
        index_path,
        row_count,
        fraction=fraction,
        seed=seed,
    )


def verify_dataset_property_manifest(
    metadata_rows,
    processing_manifest_path=PROCESSING_MANIFEST_PATH,
    target_schema_path=TARGET_SCHEMA_PATH,
):
    """Verify once that preprocessing already removed undersized variants.

    The trainer does not repeatedly screen predictor heads by sample count.
    Instead, this preflight audit checks the frozen dataset invariant encoded
    in ``processing_manifest.json`` and fails before GPU work if any retained
    dataset/sample/property/note/unit variant violates it.
    """
    with Path(processing_manifest_path).open() as file:
        manifest = json.load(file)
    with Path(target_schema_path).open() as file:
        schema = json.load(file)
    required_minimum = int(manifest.get("minimum_property_variant_count", -1))
    if required_minimum != minimum_property_samples:
        raise ValueError(
            "Dataset manifest minimum_property_variant_count does not match "
            f"training policy: {required_minimum} != {minimum_property_samples}"
        )
    allowed_targets = {
        (definition["sample"], definition["label"], definition["unit"])
        for definition in schema["target_properties"]
    }
    counts = Counter()
    for row in metadata_rows:
        dataset = _dataset_label(row)
        sample = _single_metadata_value(row, "Sample")
        for item in row:
            parsed = _split_target_item(item)
            if parsed is None:
                continue
            name, _value, unit, note = parsed
            if (sample, name, unit) not in allowed_targets:
                continue
            counts[(dataset, sample, name, _canonical_target_note(name, note), unit)] += 1
    violations = {
        key: count for key, count in counts.items() if count < required_minimum
    }
    if violations:
        preview = "; ".join(
            f"{key}={count}" for key, count in sorted(violations.items())[:10]
        )
        raise ValueError(
            "Dataset preprocessing manifest is violated by retained property "
            f"variants below {required_minimum}: {preview}"
        )
    return {
        "processing_profile": manifest.get("profile"),
        "minimum_property_variant_count": required_minimum,
        "retained_property_variant_count": len(counts),
        "verification": "single_pretraining_manifest_audit",
        "predictor_head_selection": "at_least_one_training_label",
    }


def load_property_targets(
    metadata_rows,
    report_path,
    train_rows,
    normalization_indices,
    minimum_samples=None,
):
    """Build sparse dataset/property heads normalized on optimization rows.

    Every eligible dataset/property/note/unit definition has its own scalar
    linear head. A row containing several measurement methods for the same
    property can supervise every eligible method without merging or priority
    selection. Eligibility is based on labels available in optimization rows.
    """
    if minimum_samples is not None and minimum_samples < 1:
        raise ValueError("Minimum property samples must be at least one")
    with Path(report_path).open() as file:
        report = json.load(file)

    if "partition_statistics" in report and "partitions" in report:
        report_partition = report["partition_statistics"]["train"]
        report_rows = report["row_count"]
        base_targets = list(report_partition["target_counts"])
        report_target_units = report_partition["target_units"]
        indexed_report = True
    elif "train" in report:
        report_partition = report["train"]
        report_rows = report_partition["rows"]
        base_targets = list(report_partition["target_counts"])
        report_target_units = report_partition["target_units"]
        indexed_report = False
    else:
        # Backward-compatible input for small unit tests and legacy reports.
        report_partition = report
        report_rows = report["rows"]
        base_targets = list(report["target_counts"])
        report_target_units = report["target_units"]
        indexed_report = False

    if len(metadata_rows) != report_rows:
        raise ValueError(
            f"Metadata rows ({len(metadata_rows)}) do not match report rows "
            f"({report_rows})"
        )
    if indexed_report:
        if train_rows != report["row_count"]:
            raise ValueError(
                f"Aligned rows {train_rows} do not match indexed report rows "
                f"{report['row_count']}"
            )
    elif "train" in report:
        if train_rows != report_rows:
            raise ValueError(
                f"Training rows {train_rows} do not match report rows {report_rows}"
            )
    elif train_rows != report["split"]["test_start_index"]:
        raise ValueError(
            f"Training split {train_rows} does not match report test start "
            f"{report['split']['test_start_index']}"
        )
    normalization_indices = np.asarray(normalization_indices, dtype=np.int64)
    if (
        normalization_indices.ndim != 1
        or normalization_indices.size == 0
        or normalization_indices.min() < 0
        or normalization_indices.max() >= train_rows
        or np.unique(normalization_indices).size != normalization_indices.size
    ):
        raise ValueError("Property normalization indices are invalid")

    normalization_mask = np.zeros(len(metadata_rows), dtype=bool)
    normalization_mask[normalization_indices] = True

    wet_soil_evidence = {}
    for row_index in normalization_indices:
        row = metadata_rows[int(row_index)]
        dataset = _dataset_label(row)
        evidence = wet_soil_evidence.setdefault(
            dataset,
            {"is_soil": False, "positive_moisture_labels": 0},
        )
        evidence["is_soil"] |= any(item == "Sample : Soil" for item in row)
        for item in row:
            parsed = _split_target_item(item)
            if parsed is None:
                continue
            name, value, _unit, _condition = parsed
            if name == wet_soil_moisture_target and value > 0:
                evidence["positive_moisture_labels"] += 1
    wet_soil_datasets = {
        dataset
        for dataset, evidence in wet_soil_evidence.items()
        if evidence["is_soil"] and evidence["positive_moisture_labels"] > 0
    }

    selected_targets = set(base_targets)
    canonical_units = {
        name: set(units) for name, units in report_target_units.items()
    }

    def row_values(row):
        dataset = _dataset_label(row)
        accumulated = {}
        for item in row:
            parsed = _split_target_item(item)
            if parsed is None:
                continue
            name, value, unit, condition = parsed
            if name not in selected_targets or unit not in canonical_units[name]:
                continue
            condition = _canonical_target_note(name, condition)
            key = (dataset, name, condition, unit)
            total, repeats = accumulated.get(key, (0.0, 0))
            accumulated[key] = (total + value, repeats + 1)
        return {
            key: total / repeats
            for key, (total, repeats) in accumulated.items()
        }

    raw_values_by_row = [row_values(row) for row in metadata_rows]
    filtered_values_by_row = []
    wet_soil_excluded_counts = {}
    wet_soil_excluded_optimization_counts = {}
    for row_index, values in enumerate(raw_values_by_row):
        filtered = {}
        for key, value in values.items():
            dataset, name, _condition, _unit = key
            if dataset in wet_soil_datasets and name != wet_soil_moisture_target:
                wet_soil_excluded_counts[key] = (
                    wet_soil_excluded_counts.get(key, 0) + 1
                )
                if normalization_mask[row_index]:
                    wet_soil_excluded_optimization_counts[key] = (
                        wet_soil_excluded_optimization_counts.get(key, 0) + 1
                    )
                continue
            filtered[key] = value
        filtered_values_by_row.append(filtered)

    statistics = {}
    candidate_counts = {}
    for row_index, values in enumerate(filtered_values_by_row):
        for key in values:
            candidate_counts[key] = candidate_counts.get(key, 0) + 1
        if not normalization_mask[row_index]:
            continue
        for key, value in values.items():
            total, squared_total, count = statistics.get(key, (0.0, 0.0, 0))
            statistics[key] = (
                total + value,
                squared_total + value * value,
                count + 1,
            )

    base_target_rank = {name: rank for rank, name in enumerate(base_targets)}
    ph_condition_rank = {
        condition: rank
        for rank, condition in enumerate(PH_CONDITION_DISPLAY_ORDER)
    }

    def head_sort_key(key):
        return (
            key[0],
            base_target_rank[key[1]],
            ph_condition_rank.get(key[2], len(PH_CONDITION_DISPLAY_ORDER)),
            key[2],
            key[3],
        )

    candidate_head_keys = sorted(candidate_counts, key=head_sort_key)
    predictor_training_minimum = 1 if minimum_samples is None else minimum_samples
    head_keys = [
        key
        for key in candidate_head_keys
        if statistics.get(key, (0.0, 0.0, 0))[2] >= predictor_training_minimum
    ]
    eligible_head_keys = set(head_keys)
    excluded_head_keys = [
        key for key in candidate_head_keys if key not in eligible_head_keys
    ]
    if not head_keys:
        raise ValueError(
            "No dataset/property labels meet the minimum sample requirement"
        )
    head_index = {key: index for index, key in enumerate(head_keys)}

    target_mean = np.asarray(
        [statistics[key][0] / statistics[key][2] for key in head_keys],
        dtype=np.float32,
    )
    target_std = np.ones(len(head_keys), dtype=np.float32)
    train_counts = np.asarray(
        [statistics[key][2] for key in head_keys],
        dtype=np.int64,
    )
    for index, key in enumerate(head_keys):
        total, squared_total, count = statistics[key]
        variance = max(0.0, squared_total / count - (total / count) ** 2)
        standard_deviation = math.sqrt(variance)
        if standard_deviation > 1e-8:
            target_std[index] = standard_deviation

    max_targets_per_row = max(
        (
            sum(key in head_index for key in values)
            for values in filtered_values_by_row
        ),
        default=0,
    )
    target_head_indices = np.zeros(
        (len(metadata_rows), max_targets_per_row),
        dtype=np.int64,
    )
    normalized_targets = np.zeros(
        (len(metadata_rows), max_targets_per_row),
        dtype=np.float32,
    )
    target_mask = np.zeros(
        (len(metadata_rows), max_targets_per_row),
        dtype=bool,
    )
    for row_index, values in enumerate(filtered_values_by_row):
        slot = 0
        for key, value in values.items():
            index = head_index.get(key)
            if index is None:
                continue
            target_head_indices[row_index, slot] = index
            normalized_targets[row_index, slot] = (
                value - target_mean[index]
            ) / target_std[index]
            target_mask[row_index, slot] = True
            slot += 1

    target_heads = []
    target_names = []
    target_units = {}
    for index, (dataset, name, condition, unit) in enumerate(head_keys):
        display_name = f"{dataset} | {name}"
        if condition:
            display_name += f" | {condition}"
        target_names.append(display_name)
        target_units[display_name] = unit
        target_heads.append(
            {
                "index": index,
                "dataset": dataset,
                "property": name,
                "condition": condition or None,
                "note": condition or None,
                "unit": unit,
                "name": display_name,
            }
        )

    excluded_target_heads = []
    for dataset, name, condition, unit in excluded_head_keys:
        key = (dataset, name, condition, unit)
        optimization_count = statistics.get(key, (0.0, 0.0, 0))[2]
        display_name = f"{dataset} | {name}"
        if condition:
            display_name += f" | {condition}"
        excluded_target_heads.append(
            {
                "dataset": dataset,
                "property": name,
                "condition": condition or None,
                "note": condition or None,
                "unit": unit,
                "name": display_name,
                "optimization_count": optimization_count,
                "all_training_rows_count": candidate_counts[key],
                "reason": (
                    "no_optimization_labels"
                    if optimization_count == 0
                    else "optimization_count_below_explicit_test_minimum"
                ),
            }
        )
    for dataset, name, condition, unit in sorted(
        wet_soil_excluded_counts,
        key=head_sort_key,
    ):
        display_name = f"{dataset} | {name}"
        if condition:
            display_name += f" | {condition}"
        key = (dataset, name, condition, unit)
        excluded_target_heads.append(
            {
                "dataset": dataset,
                "property": name,
                "condition": condition or None,
                "note": condition or None,
                "unit": unit,
                "name": display_name,
                "optimization_count": wet_soil_excluded_optimization_counts.get(
                    key, 0
                ),
                "all_training_rows_count": wet_soil_excluded_counts[key],
                "reason": "wet_soil_non_moisture_property",
            }
        )

    metadata = {
        "target_names": target_names,
        "target_heads": target_heads,
        "target_units": target_units,
        "target_mean": target_mean,
        "target_std": target_std,
        "train_counts": train_counts,
        "minimum_samples_per_head": predictor_training_minimum,
        "head_selection_policy": "at_least_one_training_label",
        "dataset_variant_minimum_verified_separately": minimum_property_samples,
        "excluded_target_heads": excluded_target_heads,
        "excluded_target_head_count": len(excluded_target_heads),
        "wet_soil_constraint": {
            "classification": (
                "soil dataset with at least one positive Water content value in %"
            ),
            "allowed_property": wet_soil_moisture_target,
            "datasets": sorted(wet_soil_datasets),
            "excluded_head_count": len(wet_soil_excluded_counts),
        },
        "max_targets_per_row": max_targets_per_row,
        "property_notes_are_separate": True,
        "target_head_identity": ["dataset", "property", "note", "unit"],
        "ph_conditions_are_separate": True,
        "ph_conditions": sorted(
            {key[2] for key in head_keys if key[1] == "pH"},
            key=lambda condition: (
                ph_condition_rank.get(condition, len(PH_CONDITION_DISPLAY_ORDER)),
                condition,
            ),
        ),
    }
    return target_head_indices, normalized_targets, target_mask, metadata


class SpectraDataset(Dataset):
    def __init__(
        self,
        spectra_path,
        encoded_context,
        target_head_indices,
        targets,
        target_mask,
        bandnum,
        bandwidth,
        train_rows,
    ):
        with h5py.File(spectra_path, "r") as hf:
            spectra_rows, wavelength_count = hf["spectra"].shape
            partition = hf.attrs.get("partition", "train")
            wavelengths = np.asarray(hf["wavelength"], dtype=np.int64)
        if wavelength_count != bandnum * bandwidth + 1:
            raise ValueError(
                f"Expected {bandnum * bandwidth + 1} wavelengths, got {wavelength_count}"
            )
        if not (
            spectra_rows
            == encoded_context.shape[0]
            == target_head_indices.shape[0]
            == targets.shape[0]
            == target_mask.shape[0]
        ):
            raise ValueError("Spectra, context, and target row counts do not match")
        if spectra_rows != train_rows:
            raise ValueError(
                f"HDF5 rows {spectra_rows} do not match training rows {train_rows}"
            )
        if partition not in {"train", "all", "indexed"}:
            raise ValueError(
                f"Expected the unified indexed HDF5, found partition {partition!r}"
            )
        expected_wavelengths = np.arange(350, 2501, dtype=np.int64)
        if not np.array_equal(wavelengths, expected_wavelengths):
            raise ValueError("Training wavelengths must be consecutive 350--2500 nm")

        # Shuffled row-by-row reads are pathological for the HDF5 files because
        # each compressed chunk contains 512 complete spectra. Load the training
        # split once so DataLoader workers share it through copy-on-write memory
        # and preserve the same randomized sampling without repeated decompression.
        with h5py.File(spectra_path, "r") as hf:
            self.spectra = hf["spectra"][:]
        self.props = encoded_context

        self.length = train_rows
        self.bandnum = bandnum
        self.bandwidth = bandwidth
        self.target_head_indices = target_head_indices
        self.targets = targets
        self.target_mask = target_mask

    def __getitem__(self, idx):
        spectrum = self.spectra[idx]
        spectra = spectrum[:-1].reshape(self.bandnum, self.bandwidth)
        spectrum_2500 = spectrum[-1]
        props = self.props[idx]
        return (
            torch.from_numpy(props).float(),
            torch.from_numpy(spectra).float(),
            torch.tensor(spectrum_2500, dtype=torch.float32),
            torch.from_numpy(self.target_head_indices[idx]),
            torch.from_numpy(self.targets[idx]),
            torch.from_numpy(self.target_mask[idx]),
        )

    def __len__(self):
        return self.length


class DatasetPropertyLinearPredictor(nn.Module):
    """Independent head for every dataset/property/note/unit definition."""

    def __init__(self, wavelength_count, target_count):
        super().__init__()
        # Standardized targets have zero training mean. Starting every head at
        # that mean avoids injecting arbitrary random spectral gradients when
        # the delayed predictor objective first activates.
        self.weight = nn.Parameter(torch.zeros(target_count, wavelength_count))
        self.bias = nn.Parameter(torch.zeros(target_count))

    def forward(self, spectra, target_head_indices):
        selected_weight = F.embedding(target_head_indices, self.weight)
        selected_bias = F.embedding(
            target_head_indices,
            self.bias.unsqueeze(1),
        ).squeeze(-1)
        return torch.einsum("bsw,bw->bs", selected_weight, spectra) + selected_bias

    def forward_pairs(self, spectra, row_indices, target_head_indices):
        """Predict only valid row/head pairs instead of all padded slots."""
        selected_spectra = spectra.index_select(0, row_indices)
        selected_weight = self.weight.index_select(0, target_head_indices)
        selected_bias = self.bias.index_select(0, target_head_indices)
        return (selected_spectra * selected_weight).sum(dim=1) + selected_bias


def build_property_spectra(predicted_spectra, original_2500):
    """Use the raw PadSpectra output and append measured/fitted 2500 nm."""
    predicted = predicted_spectra.reshape(predicted_spectra.shape[0], -1)

    # Per-spectrum least-squares line through 2495--2499 nm, evaluated at 2500 nm.
    fit_weights = predicted.new_tensor((-0.4, -0.1, 0.2, 0.5, 0.8))
    fitted_2500 = (predicted[:, -5:] * fit_weights).sum(dim=1).clamp_min(0)
    predicted_2500 = torch.where(
        original_2500 > 0,
        original_2500.to(predicted.dtype),
        fitted_2500,
    )
    return torch.cat((predicted, predicted_2500.unsqueeze(1)), dim=1)


def snv_transform(spectra, epsilon=property_snv_epsilon):
    """Standard normal variate transform independently for every spectrum."""
    spectra = spectra.float()
    mean = spectra.mean(dim=1, keepdim=True)
    standard_deviation = spectra.std(dim=1, keepdim=True, unbiased=False)
    return (spectra - mean) / standard_deviation.clamp_min(epsilon)


def masked_property_loss(
    prediction,
    target,
    mask,
    target_head_indices,
    target_count,
):
    """Average Smooth L1 per available head so common heads do not dominate."""
    if not mask.any():
        return prediction.sum() * 0
    valid_losses = F.smooth_l1_loss(
        prediction[mask].float(),
        target[mask].float(),
        reduction="none",
    )
    valid_heads = target_head_indices[mask]
    loss_sum = valid_losses.new_zeros(target_count)
    loss_count = valid_losses.new_zeros(target_count)
    loss_sum.scatter_add_(0, valid_heads, valid_losses)
    loss_count.scatter_add_(0, valid_heads, torch.ones_like(valid_losses))
    available_targets = loss_count > 0
    return (loss_sum[available_targets] / loss_count[available_targets]).mean()


def predict_valid_property_pairs(
    predictor,
    spectra,
    target_head_indices,
    targets,
    mask,
):
    """Return flat predictions and targets only for supervised sparse slots."""
    row_indices, target_slots = mask.nonzero(as_tuple=True)
    valid_heads = target_head_indices[row_indices, target_slots]
    valid_targets = targets[row_indices, target_slots]
    prediction = predictor.forward_pairs(spectra, row_indices, valid_heads)
    return prediction, valid_targets, valid_heads


def property_pair_loss(prediction, target, target_head_indices, target_count):
    """Average Smooth L1 per represented head for already-filtered pairs."""
    if prediction.numel() == 0:
        return prediction.sum() * 0
    valid_losses = F.smooth_l1_loss(
        prediction.float(),
        target.float(),
        reduction="none",
    )
    loss_sum = valid_losses.new_zeros(target_count)
    loss_count = valid_losses.new_zeros(target_count)
    loss_sum.scatter_add_(0, target_head_indices, valid_losses)
    loss_count.scatter_add_(0, target_head_indices, torch.ones_like(valid_losses))
    represented_heads = loss_count > 0
    return (loss_sum[represented_heads] / loss_count[represented_heads]).mean()


def count_target_labels(target_head_indices, target_mask, row_indices, target_count):
    """Count sparse target slots per head for a selected set of rows."""
    selected_heads = target_head_indices[row_indices]
    selected_mask = target_mask[row_indices]
    return np.bincount(
        selected_heads[selected_mask],
        minlength=target_count,
    ).astype(np.int64)


def dense_property_targets(
    target_head_indices,
    normalized_targets,
    target_mask,
    target_metadata,
    row_indices,
):
    """Densify selected sparse target rows in their original physical units."""
    row_indices = np.asarray(row_indices, dtype=np.int64)
    selected_heads = target_head_indices[row_indices]
    selected_targets = normalized_targets[row_indices]
    selected_mask = target_mask[row_indices]
    target_count = len(target_metadata["target_names"])
    dense_values = np.zeros((len(row_indices), target_count), dtype=np.float32)
    dense_mask = np.zeros((len(row_indices), target_count), dtype=bool)
    row_slots = np.broadcast_to(
        np.arange(len(row_indices), dtype=np.int64)[:, None],
        selected_heads.shape,
    )
    valid_rows = row_slots[selected_mask]
    valid_heads = selected_heads[selected_mask]
    flat_slots = valid_rows * target_count + valid_heads
    if np.unique(flat_slots).size != flat_slots.size:
        raise ValueError("Sparse property targets contain duplicate row/head slots")
    target_mean = np.asarray(target_metadata["target_mean"], dtype=np.float32)
    target_std = np.asarray(target_metadata["target_std"], dtype=np.float32)
    dense_values[valid_rows, valid_heads] = (
        selected_targets[selected_mask] * target_std[valid_heads]
        + target_mean[valid_heads]
    )
    dense_mask[valid_rows, valid_heads] = True
    return dense_values, dense_mask


def weights_init(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)


def update_ema(ema_model, model, decay):
    with torch.no_grad():
        for ema_param, model_param in zip(ema_model.parameters(), model.parameters()):
            ema_param.mul_(decay).add_(model_param, alpha=1 - decay)
        for ema_buffer, model_buffer in zip(ema_model.buffers(), model.buffers()):
            ema_buffer.copy_(model_buffer)


def should_update_ema(completed_optimizer_steps):
    """Return whether EMA is due after this successful optimizer step."""
    if completed_optimizer_steps < 1:
        raise ValueError("Completed optimizer steps must be positive")
    return completed_optimizer_steps % ema_update_interval == 0


def make_training_generators():
    """Create independent reproducible streams for each stochastic decision."""
    return {
        name: torch.Generator().manual_seed(seed)
        for name, seed in TRAINING_GENERATOR_SEEDS.items()
    }


def training_generator_state_dict(generators):
    if set(generators) != set(TRAINING_GENERATOR_SEEDS):
        raise ValueError("Training generator names do not match the configured streams")
    return {name: generator.get_state() for name, generator in generators.items()}


def restore_training_generator_state(generators, states):
    if set(states) != set(TRAINING_GENERATOR_SEEDS):
        raise ValueError("Checkpoint stochastic streams do not match this trainer")
    for name, generator in generators.items():
        generator.set_state(states[name].cpu())


def should_save_ema_snapshot(epoch):
    """Save standalone EMA inference weights every 1,000 epochs and at end."""
    if epoch < 0 or epoch > num_epochs:
        raise ValueError("EMA snapshot epoch is outside the training schedule")
    return epoch == num_epochs or (
        epoch > 0 and epoch % ema_checkpoint_interval == 0
    )


def prepare_ema_model(model):
    ema_model = copy.deepcopy(model)
    ema_model.requires_grad_(False)
    ema_model.eval()
    return ema_model


class MAELoss(nn.Module):
    def __init__(self, bandnum=43, bandwidth=50):
        super().__init__()
        self.bandnum = bandnum
        self.bandwidth = bandwidth

    @staticmethod
    def masked_sum_and_count(values, mask):
        return values.masked_fill(~mask, 0).sum(), mask.sum()

    @staticmethod
    def _require_nonempty(count, name):
        # Never turn a missing supervision mask into a plausible zero loss.
        if int(count.detach().item()) == 0:
            raise RuntimeError(
                f"{name} contains zero points. The unmasked target spectra must "
                "be passed to MAELoss together with the model's missing-band mask."
            )

    def components(self, real_spectra, padspectra, mask_pad):
        """Return an FP32 objective plus independently auditable components."""
        if real_spectra.shape != padspectra.shape:
            raise ValueError(
                "Target and predicted spectra must have identical shapes; got "
                f"{tuple(real_spectra.shape)} and {tuple(padspectra.shape)}"
            )
        if real_spectra.ndim != 3 or real_spectra.shape[1:] != (
            self.bandnum,
            self.bandwidth,
        ):
            raise ValueError(
                f"Spectra must have shape (batch, {self.bandnum}, "
                f"{self.bandwidth})"
            )
        if mask_pad.ndim != 3 or mask_pad.shape[:2] != real_spectra.shape[:2]:
            raise ValueError("The missing-band mask does not match the spectra")
        if mask_pad.shape[2] not in (1, self.bandwidth):
            raise ValueError("The missing-band mask must have width 1 or bandwidth")

        # Loss arithmetic stays in FP32 even while the model itself uses AMP.
        target = real_spectra.float()
        prediction = padspectra.float()
        missing_mask = mask_pad.to(torch.bool).expand_as(target)
        reconstruction_mask = missing_mask & torch.isfinite(target) & (target > 0)
        reconstruction_sum, reconstruction_points = self.masked_sum_and_count(
            torch.abs(target - prediction),
            reconstruction_mask,
        )
        reconstruction_squared_error_sum, _ = self.masked_sum_and_count(
            (target - prediction).square(),
            reconstruction_mask,
        )
        self._require_nonempty(reconstruction_points, "Reconstruction mask")
        reconstruction_mae = reconstruction_sum / reconstruction_points.to(
            reconstruction_sum.dtype
        )

        completed_spectra = torch.where(missing_mask, prediction, target).reshape(
            target.shape[0], self.bandnum * self.bandwidth
        )
        mask_flat = missing_mask.reshape(missing_mask.size(0), -1)
        mask_smooth = mask_flat[:, :-1] | mask_flat[:, 1:]
        smooth_sum, smooth_points = self.masked_sum_and_count(
            (completed_spectra[:, :-1] - completed_spectra[:, 1:]).square(),
            mask_smooth,
        )
        self._require_nonempty(smooth_points, "Smoothness mask")
        tv_loss = smooth_sum / smooth_points.to(smooth_sum.dtype)

        negative_sum, negative_points = self.masked_sum_and_count(
            torch.clamp(-completed_spectra, min=0),
            mask_flat,
        )
        self._require_nonempty(negative_points, "Negative-value mask")
        negative_loss = negative_sum / negative_points.to(negative_sum.dtype)
        objective = reconstruction_mae + tv_loss * 2 + negative_loss
        return {
            "objective": objective,
            "reconstruction_mae": reconstruction_mae,
            "reconstruction_absolute_error_sum": reconstruction_sum,
            "reconstruction_squared_error_sum": reconstruction_squared_error_sum,
            "reconstruction_points": reconstruction_points,
            "tv": tv_loss,
            "negative": negative_loss,
        }

    def forward(self, real_spectra, padspectra, mask_pad):
        return self.components(real_spectra, padspectra, mask_pad)["objective"]


num_workers = 4


def build_augmented_batch(
    spectra_batch,
    epoch,
    batch_index,
    generator=None,
):
    all0real = (spectra_batch == 0).sum()
    non_zero_mask = torch.ones(
        spectra_batch.size(0),
        dtype=torch.bool,
        device=spectra_batch.device,
    )
    spectra_batch_input = spectra_batch

    for _ in range(16):
        spectra_batch_input = spectra_batch.clone()
        large_band_limit = 30 if epoch >= 1000 else 25
        L = int(
            torch.randint(
                1,
                large_band_limit,
                (),
                generator=generator,
            ).item()
        )
        randstop = int(torch.randint(0, 3, (), generator=generator).item())

        if batch_index % 2 == 0:
            spectra_batch_input[:, :L, :] = 0
            if randstop == 2:
                S = int(torch.randint(1, 7, (), generator=generator).item())
                spectra_batch_input[:, -S:, :] = 0
        else:
            spectra_batch_input[:, -L:, :] = 0
            if randstop == 2:
                S = int(torch.randint(1, 7, (), generator=generator).item())
                spectra_batch_input[:, :S, :] = 0

        if batch_index % 7 == 0:
            mid0preidx = int(
                torch.randint(
                    0,
                    bandnum - 3,
                    (),
                    generator=generator,
                ).item()
            )
            mid0bands = int(torch.randint(1, 9, (), generator=generator).item())
            mid_end = min(bandnum, mid0preidx + mid0bands)
            spectra_batch_input[:, mid0preidx:mid_end, :] = 0

        zero_mask_input = spectra_batch_input == 0
        non_zero_mask = usable_spectrum_mask(spectra_batch_input)
        all0inbatch = zero_mask_input.sum()
        if non_zero_mask.any() and all0inbatch > all0real:
            break

    return spectra_batch_input, non_zero_mask


def usable_spectrum_mask(spectra):
    """Return rows with at least eight fully observed 50 nm bands."""
    if spectra.ndim != 3 or spectra.shape[1:] != (bandnum, bandwidth):
        raise ValueError(
            f"Spectra must have shape (batch, {bandnum}, {bandwidth})"
        )
    missing_bands = (spectra == 0).any(dim=-1).sum(dim=-1)
    return missing_bands <= maximum_missing_bands


def move_batch_to_device(batch):
    (
        prop_batch,
        spectra_batch,
        spectrum_2500_batch,
        property_head_index_batch,
        property_target_batch,
        property_mask_batch,
    ) = batch
    return (
        prop_batch.to(device, non_blocking=True),
        spectra_batch.to(device, non_blocking=True),
        spectrum_2500_batch.to(device, non_blocking=True),
        property_head_index_batch.to(device, non_blocking=True),
        property_target_batch.to(device, non_blocking=True),
        property_mask_batch.to(device, non_blocking=True),
    )


def property_context_slots(target_metadata):
    """Map each predictor head to its matching numerical context slot."""
    slots = []
    for head in target_metadata["target_heads"]:
        definition = (head["property"], head.get("condition"), head.get("unit"))
        if definition[:2] == ("pH", "Water"):
            slot = CONTEXT_SLOT_NAMES.index("ph_in_water")
        elif definition[:2] == ("pH", "CaCl2 solution"):
            slot = CONTEXT_SLOT_NAMES.index("ph_in_cacl2")
        elif definition[0] == "pH":
            # Every other pH method is stripped to the generic pH input slot,
            # while its original note remains a separate predictor head.
            slot = CONTEXT_SLOT_NAMES.index("ph_generic")
        elif definition == ("Water content", None, "g/m2"):
            slot = CONTEXT_SLOT_NAMES.index("leaf_water")
        elif definition == ("Leaf mass per area", None, "g/m2"):
            slot = CONTEXT_SLOT_NAMES.index("leaf_lma")
        elif definition == (wet_soil_moisture_target, None, "%"):
            slot = CONTEXT_SLOT_NAMES.index("soil_gravimetric_water")
        else:
            slot = -1
        slots.append(slot)
    return torch.as_tensor(
        slots,
        dtype=torch.long,
        device=device,
    )


def remove_conditioned_property_targets(
    property_mask,
    property_head_indices,
    conditioned_context,
    context_slot_by_head,
    pad_index,
):
    """Prevent a supplied numerical condition from supervising itself."""
    if context_slot_by_head.numel() == 0:
        return property_mask
    if context_slot_by_head.device != property_head_indices.device:
        context_slot_by_head = context_slot_by_head.to(property_head_indices.device)
    target_slots = context_slot_by_head[property_head_indices]
    has_conditioning_slot = target_slots >= 0
    safe_slots = target_slots.clamp_min(0)
    supplied_as_context = torch.gather(
        conditioned_context[:, :, 0],
        1,
        safe_slots,
    ) != pad_index
    return property_mask & ~(has_conditioning_slot & supplied_as_context)


def evaluate_validation(
    aem,
    property_predictor,
    loader,
    criterion,
    epoch,
    target_metadata,
    context_metadata,
    context_slot_by_head,
):
    """Evaluate EMA weights with a fixed spectral mask and full given context."""
    aem.eval()
    if property_predictor is not None:
        property_predictor.eval()
    property_loss_active = (
        PROPERTY_CONSTRAINT_ENABLED and epoch >= property_start_epoch
    )

    spectral_objective_sum = torch.zeros((), dtype=torch.float64, device=device)
    reconstruction_absolute_error_sum = torch.zeros_like(spectral_objective_sum)
    reconstruction_squared_error_sum = torch.zeros_like(spectral_objective_sum)
    reconstruction_points = torch.zeros((), dtype=torch.int64, device=device)
    tv_sum = torch.zeros_like(spectral_objective_sum)
    negative_sum = torch.zeros_like(spectral_objective_sum)
    property_sum = torch.zeros_like(spectral_objective_sum)
    property_batches = 0
    target_count = len(target_metadata["target_names"])
    property_absolute_error = torch.zeros(
        target_count,
        dtype=torch.float64,
        device=device,
    )
    property_squared_error = torch.zeros_like(property_absolute_error)
    property_valid_counts = torch.zeros_like(property_absolute_error)
    target_std = torch.as_tensor(
        target_metadata["target_std"],
        dtype=torch.float32,
        device=device,
    )

    rng_devices = [torch.cuda.current_device()] if amp_enabled else []
    with torch.random.fork_rng(devices=rng_devices):
        torch.manual_seed(validation_mask_seed)
        if amp_enabled:
            torch.cuda.manual_seed(validation_mask_seed)
        generator = torch.Generator().manual_seed(validation_mask_seed)

        with torch.inference_mode():
            for batch_index, batch in enumerate(loader):
                (
                    prop_batch,
                    spectra_batch,
                    spectrum_2500_batch,
                    property_head_index_batch,
                    property_target_batch,
                    property_mask_batch,
                ) = move_batch_to_device(batch)
                spectra_batch_input, non_zero_mask = build_augmented_batch(
                    spectra_batch,
                    validation_mask_epoch,
                    batch_index,
                    generator=generator,
                )
                prop_batch, _conditioning_modes = apply_conditioning_schedule(
                    prop_batch,
                    context_metadata["pad_index"],
                    fixed_mode=TRAINING_CONDITIONING_MODE_INDEX,
                    fixed_mode_measurement_dropout=False,
                    mask_optional_context=False,
                )

                with autocast(device_type=device.type, enabled=amp_enabled):
                    padspectra, pad_mask = aem(spectra_batch_input, prop_batch)

                # Keep the reconstruction objective outside autocast and retain
                # its numerator/count so validation cannot accidentally report
                # only the tiny TV regularizer as the spectral metric.
                spectral_components = criterion.components(
                    spectra_batch[non_zero_mask],
                    padspectra[non_zero_mask],
                    pad_mask[non_zero_mask],
                )
                spectral_objective = spectral_components["objective"]

                property_loss = spectral_objective.new_zeros(())
                with autocast(device_type=device.type, enabled=amp_enabled):
                    if property_loss_active:
                        property_mask_batch = remove_conditioned_property_targets(
                            property_mask_batch,
                            property_head_index_batch,
                            prop_batch,
                            context_slot_by_head,
                            context_metadata["pad_index"],
                        )
                        usable_property_mask = (
                            property_mask_batch & non_zero_mask.unsqueeze(1)
                        )
                        property_spectra = build_property_spectra(
                            padspectra,
                            spectrum_2500_batch,
                        )
                        property_spectra = snv_transform(property_spectra)
                        (
                            property_prediction,
                            valid_property_target,
                            valid_heads,
                        ) = predict_valid_property_pairs(
                            property_predictor,
                            property_spectra,
                            property_head_index_batch,
                            property_target_batch,
                            usable_property_mask,
                        )
                        property_loss = property_pair_loss(
                            property_prediction,
                            valid_property_target.to(property_prediction.dtype),
                            valid_heads,
                            target_count,
                        )
                        normalized_error = (
                            property_prediction.float()
                            - valid_property_target.float()
                        )
                        valid_error = normalized_error * target_std[valid_heads]
                        property_absolute_error.scatter_add_(
                            0,
                            valid_heads,
                            valid_error.abs().to(torch.float64),
                        )
                        property_squared_error.scatter_add_(
                            0,
                            valid_heads,
                            valid_error.square().to(torch.float64),
                        )
                        property_valid_counts.scatter_add_(
                            0,
                            valid_heads,
                            torch.ones_like(valid_error, dtype=torch.float64),
                        )
                        property_sum.add_(property_loss.to(torch.float64))
                        if valid_heads.numel() > 0:
                            property_batches += 1

                spectral_objective_sum.add_(
                    spectral_objective.to(torch.float64)
                )
                reconstruction_absolute_error_sum.add_(
                    spectral_components[
                        "reconstruction_absolute_error_sum"
                    ].to(torch.float64)
                )
                reconstruction_squared_error_sum.add_(
                    spectral_components[
                        "reconstruction_squared_error_sum"
                    ].to(torch.float64)
                )
                reconstruction_points.add_(
                    spectral_components["reconstruction_points"].to(torch.int64)
                )
                tv_sum.add_(spectral_components["tv"].to(torch.float64))
                negative_sum.add_(
                    spectral_components["negative"].to(torch.float64)
                )

    property_mae = {}
    property_rmse = {}
    property_metric_counts = {}
    property_absolute_error = property_absolute_error.cpu().numpy()
    property_squared_error = property_squared_error.cpu().numpy()
    property_valid_counts = property_valid_counts.cpu().numpy().astype(np.int64)
    for target_index, target_name in enumerate(target_metadata["target_names"]):
        count = int(property_valid_counts[target_index])
        property_metric_counts[target_name] = count
        if count:
            property_mae[target_name] = float(
                property_absolute_error[target_index] / count
            )
            property_rmse[target_name] = float(
                math.sqrt(property_squared_error[target_index] / count)
            )
        else:
            property_mae[target_name] = None
            property_rmse[target_name] = None

    reconstruction_point_count = int(reconstruction_points.item())
    if reconstruction_point_count == 0:
        raise RuntimeError("Validation produced zero reconstruction target points")
    spectral_mae = (
        reconstruction_absolute_error_sum / reconstruction_points
    ).item()
    spectral_rmse = torch.sqrt(
        reconstruction_squared_error_sum / reconstruction_points
    ).item()
    property_average = (
        (property_sum / property_batches).item() if property_batches else 0.0
    )
    return {
        "total": spectral_mae + property_loss_weight * property_average,
        "spectral": spectral_mae,
        "spectral_rmse": spectral_rmse,
        "spectral_objective": (spectral_objective_sum / len(loader)).item(),
        "spectral_tv": (tv_sum / len(loader)).item(),
        "spectral_negative": (negative_sum / len(loader)).item(),
        "spectral_reconstruction_points": reconstruction_point_count,
        "property": property_average,
        "property_active": property_loss_active,
        "property_mae_original_units": property_mae,
        "property_rmse_original_units": property_rmse,
        "property_metric_counts": property_metric_counts,
    }


def write_json_atomic(path, payload):
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w") as file:
        json.dump(payload, file)
    temporary_path.replace(path)


def append_json_line(path, payload):
    """Append detailed validation diagnostics without rewriting prior epochs."""
    with path.open("a") as file:
        file.write(json.dumps(payload, separators=(",", ":")))
        file.write("\n")


def save_property_predictor(
    predictor,
    target_metadata,
    epoch,
    global_step,
):
    checkpoint = {
        "format_version": training_checkpoint_format_version,
        "training_variant": TRAINING_VARIANT,
        "training_run_name": TRAINING_RUN_NAME,
        "conditioning_mode": TRAINING_CONDITIONING_MODE,
        "property_constraint_enabled": PROPERTY_CONSTRAINT_ENABLED,
        "spectrometer_dropout_enabled": SPECTROMETER_DROPOUT_ENABLED,
        "spectrometer_dropout_probability": (
            TRAINING_SPECTROMETER_DROPOUT_PROBABILITY
        ),
        "measurement_mode_dropout_enabled": (
            MEASUREMENT_MODE_DROPOUT_ENABLED
        ),
        "measurement_mode_dropout_probability": (
            TRAINING_MEASUREMENT_MODE_DROPOUT_PROBABILITY
        ),
        "feed_forward_type": TRAINING_FEED_FORWARD_TYPE,
        "model_state_dict": predictor.state_dict(),
        "target_names": target_metadata["target_names"],
        "target_heads": target_metadata["target_heads"],
        "target_units": target_metadata["target_units"],
        "target_mean": target_metadata["target_mean"].tolist(),
        "target_std": target_metadata["target_std"].tolist(),
        "input_wavelength_count": bandnum * bandwidth + 1,
        "property_start_epoch": property_start_epoch,
        "property_input_spectra": property_input_spectra,
        "property_predictor_type": property_predictor_type,
        "property_predictor_initialization": "zero_standardized_target_mean",
        "property_snv_epsilon": property_snv_epsilon,
        "property_loss_weight": property_loss_weight,
        "minimum_property_samples": minimum_property_samples,
        "property_manifest_audit": target_metadata.get("property_manifest_audit"),
        "excluded_target_heads": target_metadata["excluded_target_heads"],
        "wet_soil_constraint": target_metadata.get("wet_soil_constraint"),
        "property_notes_are_separate": True,
        "target_head_identity": target_metadata["target_head_identity"],
        "ph_conditions_are_separate": True,
        "ph_conditions": target_metadata["ph_conditions"],
        "weights": "ema",
        "ema_decay": ema_decay,
        "ema_update_interval": ema_update_interval,
        "ema_checkpoint_interval": ema_checkpoint_interval,
        "periodic_ema_snapshots_enabled": PERIODIC_EMA_SNAPSHOTS_ENABLED,
        "learning_rate_schedule": "warmup_plateau_two_stage_cosine_v1",
        "warmup_steps": warmup_steps,
        "lr_plateau_end_epoch": lr_plateau_end_epoch,
        "training_end_epoch": num_epochs,
        "initial_schedule_end_epoch": initial_schedule_end_epoch,
        "continuation_final_learning_rate": continuation_final_learning_rate,
        "epoch": epoch,
        "global_step": global_step,
        "fit_2500_nm": {
            "source_wavelengths": [2495, 2496, 2497, 2498, 2499],
            "weights": [-0.4, -0.1, 0.2, 0.5, 0.8],
            "preserve_measured_2500": True,
        },
    }
    torch.save(
        checkpoint,
        WEIGHTS_DIR / f"property_predictor_linear_snv_{epoch}.pth",
    )


def save_training_checkpoint(
    checkpoint_path,
    checkpoint_type,
    aem,
    ema_model,
    property_predictor,
    ema_property_predictor,
    optimizer,
    scheduler,
    scaler,
    target_metadata,
    context_metadata,
    split_metadata,
    epoch,
    global_step,
    validation_metrics,
    history,
    best_validation_spectral,
    best_validation_combined,
    training_generators,
):
    checkpoint = {
        "format_version": training_checkpoint_format_version,
        "checkpoint_type": checkpoint_type,
        "training_variant": TRAINING_VARIANT,
        "training_run_name": TRAINING_RUN_NAME,
        "conditioning_mode": TRAINING_CONDITIONING_MODE,
        "property_constraint_enabled": PROPERTY_CONSTRAINT_ENABLED,
        "spectrometer_dropout_enabled": SPECTROMETER_DROPOUT_ENABLED,
        "spectrometer_dropout_probability": (
            TRAINING_SPECTROMETER_DROPOUT_PROBABILITY
        ),
        "measurement_mode_dropout_enabled": (
            MEASUREMENT_MODE_DROPOUT_ENABLED
        ),
        "measurement_mode_dropout_probability": (
            TRAINING_MEASUREMENT_MODE_DROPOUT_PROBABILITY
        ),
        "feed_forward_type": TRAINING_FEED_FORWARD_TYPE,
        "training_seed": TRAINING_SEED,
        "model_state_dict": aem.state_dict(),
        "ema_model_state_dict": ema_model.state_dict(),
        "property_predictor_state_dict": (
            property_predictor.state_dict()
            if property_predictor is not None
            else None
        ),
        "ema_property_predictor_state_dict": (
            ema_property_predictor.state_dict()
            if ema_property_predictor is not None
            else None
        ),
        "validation_weights": "ema",
        "validation_conditioning_policy": validation_conditioning_policy,
        "best_validation_metric": best_validation_metric,
        "secondary_validation_metric": secondary_validation_metric,
        "recommended_inference_model_key": "ema_model_state_dict",
        "recommended_inference_property_predictor_key": (
            "ema_property_predictor_state_dict"
            if PROPERTY_CONSTRAINT_ENABLED
            else None
        ),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "target_names": target_metadata["target_names"],
        "target_heads": target_metadata["target_heads"],
        "target_units": target_metadata["target_units"],
        "target_mean": target_metadata["target_mean"].tolist(),
        "target_std": target_metadata["target_std"].tolist(),
        "context": context_metadata,
        "split": split_metadata,
        "validation": validation_metrics,
        "history": history,
        "best_validation_spectral": best_validation_spectral,
        "best_validation_combined": best_validation_combined,
        "ema_decay": ema_decay,
        "ema_update_interval": ema_update_interval,
        "ema_checkpoint_interval": ema_checkpoint_interval,
        "periodic_ema_snapshots_enabled": PERIODIC_EMA_SNAPSHOTS_ENABLED,
        "minimum_observed_bands": minimum_observed_bands,
        "maximum_missing_bands": maximum_missing_bands,
        "property_start_epoch": property_start_epoch,
        "property_input_spectra": property_input_spectra,
        "property_predictor_type": property_predictor_type,
        "property_predictor_initialization": "zero_standardized_target_mean",
        "property_snv_epsilon": property_snv_epsilon,
        "property_loss_weight": property_loss_weight,
        "minimum_property_samples": minimum_property_samples,
        "property_manifest_audit": target_metadata.get("property_manifest_audit"),
        "excluded_target_heads": target_metadata["excluded_target_heads"],
        "wet_soil_constraint": target_metadata.get("wet_soil_constraint"),
        "property_notes_are_separate": True,
        "target_head_identity": target_metadata["target_head_identity"],
        "ph_conditions_are_separate": True,
        "ph_conditions": target_metadata["ph_conditions"],
        "validation_interval": validation_interval,
        "validation_mask_seed": validation_mask_seed,
        "validation_mask_epoch": validation_mask_epoch,
        "learning_rate_schedule": "warmup_plateau_two_stage_cosine_v1",
        "warmup_steps": warmup_steps,
        "lr_plateau_end_epoch": lr_plateau_end_epoch,
        "training_end_epoch": num_epochs,
        "initial_schedule_end_epoch": initial_schedule_end_epoch,
        "continuation_final_learning_rate": continuation_final_learning_rate,
        "epoch": epoch,
        "global_step": global_step,
        "training_generator_seeds": TRAINING_GENERATOR_SEEDS,
        "training_generator_states": training_generator_state_dict(
            training_generators
        ),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (torch.cuda.get_rng_state_all() if amp_enabled else None),
    }
    temporary_path = checkpoint_path.with_suffix(".pth.tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(checkpoint_path)


def load_training_checkpoint(
    checkpoint_path,
    aem,
    ema_model,
    property_predictor,
    ema_property_predictor,
    optimizer,
    scheduler,
    scaler,
    target_metadata,
    context_metadata,
    split_metadata,
    training_generators,
):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint.get("format_version") != training_checkpoint_format_version:
        raise ValueError(
            "Unsupported training checkpoint format: expected "
            f"{training_checkpoint_format_version}, found "
            f"{checkpoint.get('format_version')}. The runtime-conditioning "
            "design requires a fresh checkpoint."
        )
    if checkpoint.get("training_variant") != TRAINING_VARIANT:
        raise ValueError(
            f"Checkpoint variant {checkpoint.get('training_variant')!r} does not "
            f"match active variant {TRAINING_VARIANT!r}"
        )
    if checkpoint.get("training_run_name", TRAINING_VARIANT) != TRAINING_RUN_NAME:
        raise ValueError(
            f"Checkpoint run name "
            f"{checkpoint.get('training_run_name', TRAINING_VARIANT)!r} does not "
            f"match active run {TRAINING_RUN_NAME!r}"
        )
    if checkpoint.get("conditioning_mode") != TRAINING_CONDITIONING_MODE:
        raise ValueError("Checkpoint conditioning mode does not match this run")
    if (
        checkpoint.get("property_constraint_enabled")
        is not PROPERTY_CONSTRAINT_ENABLED
    ):
        raise ValueError("Checkpoint property-constraint setting does not match")
    if (
        checkpoint.get("spectrometer_dropout_enabled", False)
        is not SPECTROMETER_DROPOUT_ENABLED
    ):
        raise ValueError("Checkpoint spectrometer-dropout setting does not match")
    if (
        checkpoint.get(
            "spectrometer_dropout_probability",
            TRAINING_SPECTROMETER_DROPOUT_PROBABILITY,
        )
        != TRAINING_SPECTROMETER_DROPOUT_PROBABILITY
    ):
        raise ValueError(
            "Checkpoint spectrometer-dropout probability does not match"
        )
    if (
        checkpoint.get("measurement_mode_dropout_enabled", False)
        is not MEASUREMENT_MODE_DROPOUT_ENABLED
    ):
        raise ValueError("Checkpoint measurement-mode dropout setting does not match")
    if (
        checkpoint.get(
            "measurement_mode_dropout_probability",
            TRAINING_MEASUREMENT_MODE_DROPOUT_PROBABILITY,
        )
        != TRAINING_MEASUREMENT_MODE_DROPOUT_PROBABILITY
    ):
        raise ValueError(
            "Checkpoint measurement-mode dropout probability does not match"
        )
    if checkpoint.get("training_seed") != TRAINING_SEED:
        raise ValueError("Checkpoint training seed does not match this run")
    if (
        checkpoint.get("feed_forward_type", psm.FEED_FORWARD_DENSE)
        != TRAINING_FEED_FORWARD_TYPE
    ):
        raise ValueError("Checkpoint feed-forward architecture does not match")
    expected_schedule = {
        "learning_rate_schedule": "warmup_plateau_two_stage_cosine_v1",
        "warmup_steps": warmup_steps,
        "lr_plateau_end_epoch": lr_plateau_end_epoch,
        "training_end_epoch": num_epochs,
        "initial_schedule_end_epoch": initial_schedule_end_epoch,
        "continuation_final_learning_rate": continuation_final_learning_rate,
    }
    for schedule_key, expected_value in expected_schedule.items():
        if checkpoint.get(schedule_key) != expected_value:
            raise ValueError(
                f"Checkpoint {schedule_key} {checkpoint.get(schedule_key)!r} "
                f"does not match configured {expected_value!r}"
            )
    if checkpoint["target_names"] != target_metadata["target_names"]:
        raise ValueError("Checkpoint property targets do not match this dataset")
    if checkpoint.get("context") != context_metadata:
        raise ValueError("Checkpoint runtime conditioning does not match this dataset")
    checkpoint_split = checkpoint["split"]
    for hash_key in (
        "split_index_sha256",
        "training_index_sha256",
        "validation_index_sha256",
        "test_index_sha256",
    ):
        if checkpoint_split.get(hash_key) != split_metadata.get(hash_key):
            raise ValueError(f"Checkpoint split mismatch for {hash_key}")
    if checkpoint.get("ema_decay") != ema_decay:
        raise ValueError(
            f"Checkpoint EMA decay {checkpoint.get('ema_decay')} does not match "
            f"configured {ema_decay}"
        )
    if checkpoint.get("ema_update_interval") != ema_update_interval:
        raise ValueError(
            f"Checkpoint EMA update interval "
            f"{checkpoint.get('ema_update_interval')} does not match configured "
            f"{ema_update_interval}"
        )
    if checkpoint.get("ema_checkpoint_interval") != ema_checkpoint_interval:
        raise ValueError(
            f"Checkpoint EMA snapshot interval "
            f"{checkpoint.get('ema_checkpoint_interval')} does not match "
            f"configured {ema_checkpoint_interval}"
        )
    if checkpoint.get(
        "periodic_ema_snapshots_enabled",
        True,
    ) is not PERIODIC_EMA_SNAPSHOTS_ENABLED:
        raise ValueError(
            "Checkpoint periodic EMA snapshot policy does not match this run"
        )
    if checkpoint.get("minimum_observed_bands") != minimum_observed_bands:
        raise ValueError(
            f"Checkpoint minimum observed bands "
            f"{checkpoint.get('minimum_observed_bands')} does not match "
            f"configured {minimum_observed_bands}"
        )
    if checkpoint.get("maximum_missing_bands") != maximum_missing_bands:
        raise ValueError(
            f"Checkpoint maximum missing bands "
            f"{checkpoint.get('maximum_missing_bands')} does not match "
            f"configured {maximum_missing_bands}"
        )
    if checkpoint.get("property_start_epoch") != property_start_epoch:
        raise ValueError(
            f"Checkpoint property start epoch "
            f"{checkpoint.get('property_start_epoch')} does not match "
            f"configured {property_start_epoch}"
        )
    if checkpoint.get("property_input_spectra") != property_input_spectra:
        raise ValueError(
            f"Checkpoint property input "
            f"{checkpoint.get('property_input_spectra')} does not match "
            f"configured {property_input_spectra}"
        )
    if (
        PROPERTY_CONSTRAINT_ENABLED
        and checkpoint.get("property_loss_weight") != property_loss_weight
    ):
        raise ValueError(
            f"Checkpoint property loss weight "
            f"{checkpoint.get('property_loss_weight')} does not match "
            f"configured {property_loss_weight}"
        )
    if checkpoint.get("property_predictor_type") != property_predictor_type:
        raise ValueError(
            f"Checkpoint predictor type "
            f"{checkpoint.get('property_predictor_type')} does not match "
            f"configured {property_predictor_type}"
        )
    if checkpoint.get("property_snv_epsilon") != property_snv_epsilon:
        raise ValueError(
            f"Checkpoint SNV epsilon "
            f"{checkpoint.get('property_snv_epsilon')} does not match "
            f"configured {property_snv_epsilon}"
        )
    if checkpoint.get("minimum_property_samples") != minimum_property_samples:
        raise ValueError(
            f"Checkpoint minimum property samples "
            f"{checkpoint.get('minimum_property_samples')} does not match "
            f"configured {minimum_property_samples}"
        )
    if checkpoint.get("property_manifest_audit") != target_metadata.get(
        "property_manifest_audit"
    ):
        raise ValueError("Checkpoint property-manifest audit does not match")
    if checkpoint.get("excluded_target_heads") != target_metadata[
        "excluded_target_heads"
    ]:
        raise ValueError("Checkpoint excluded property heads do not match this dataset")
    if checkpoint.get("wet_soil_constraint") != target_metadata.get(
        "wet_soil_constraint"
    ):
        raise ValueError("Checkpoint wet-soil property policy does not match")
    if checkpoint.get("property_notes_are_separate") is not True:
        raise ValueError("Checkpoint does not use separate property-note heads")
    if checkpoint.get("target_head_identity") != target_metadata[
        "target_head_identity"
    ]:
        raise ValueError("Checkpoint property-head identity does not match")
    if checkpoint.get("ph_conditions_are_separate") is not True:
        raise ValueError(
            "Checkpoint does not use separate pH condition heads"
        )

    aem.load_state_dict(checkpoint["model_state_dict"])
    ema_model.load_state_dict(checkpoint["ema_model_state_dict"])
    if PROPERTY_CONSTRAINT_ENABLED:
        if property_predictor is None or ema_property_predictor is None:
            raise ValueError("Predictor-constrained checkpoint requires predictors")
        property_predictor.load_state_dict(
            checkpoint["property_predictor_state_dict"]
        )
        ema_property_predictor.load_state_dict(
            checkpoint["ema_property_predictor_state_dict"]
        )
    elif (
        checkpoint.get("property_predictor_state_dict") is not None
        or checkpoint.get("ema_property_predictor_state_dict") is not None
    ):
        raise ValueError("Pure-padding checkpoint unexpectedly contains a predictor")
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    if checkpoint.get("training_generator_seeds") != TRAINING_GENERATOR_SEEDS:
        raise ValueError("Checkpoint stochastic-stream seeds do not match")
    restore_training_generator_state(
        training_generators,
        checkpoint["training_generator_states"],
    )

    torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    if amp_enabled and checkpoint["cuda_rng_state_all"] is not None:
        torch.cuda.set_rng_state_all(
            [state.cpu() for state in checkpoint["cuda_rng_state_all"]]
        )

    checkpoint_uses_current_validation_protocol = (
        checkpoint.get("validation_conditioning_policy")
        == validation_conditioning_policy
        and checkpoint.get("best_validation_metric") == best_validation_metric
        and checkpoint.get("secondary_validation_metric")
        == secondary_validation_metric
    )
    best_spectral = (
        float(checkpoint["best_validation_spectral"])
        if checkpoint_uses_current_validation_protocol
        else float("inf")
    )
    best_combined = (
        float(checkpoint["best_validation_combined"])
        if checkpoint_uses_current_validation_protocol
        else float("inf")
    )

    return {
        "start_epoch": int(checkpoint["epoch"]) + 1,
        "global_step": int(checkpoint["global_step"]),
        "history": checkpoint["history"],
        "best_validation_spectral": best_spectral,
        "best_validation_combined": best_combined,
        "validation_protocol_reset": not checkpoint_uses_current_validation_protocol,
        "validation": checkpoint.get("validation"),
    }


def main():
    LOSS_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(TRAINING_SEED)
    np.random.seed(TRAINING_SEED)
    if amp_enabled:
        torch.cuda.manual_seed_all(TRAINING_SEED)
    training_generators = make_training_generators()

    with METADATA_PATH.open() as file:
        metadata_rows = json.load(file)
    row_count = len(metadata_rows)
    train_indices, validation_indices, split_metadata = load_predefined_split(
        metadata_rows,
        SPLIT_INDEX_PATH,
        row_count,
    )
    property_manifest_audit = verify_dataset_property_manifest(metadata_rows)
    if (
        property_manifest_audit["processing_profile"]
        != split_metadata["processing_profile"]
    ):
        raise ValueError(
            "Processing manifest and indexed split use different profiles"
        )
    encoded_context, context_metadata = build_runtime_conditioning(
        metadata_rows,
        train_indices,
    )
    if PROPERTY_CONSTRAINT_ENABLED:
        (
            target_head_indices,
            normalized_targets,
            target_mask,
            target_metadata,
        ) = load_property_targets(
            metadata_rows,
            SPLIT_INDEX_PATH,
            row_count,
            train_indices,
        )
        target_metadata["property_manifest_audit"] = property_manifest_audit
        target_metadata["validation_counts"] = count_target_labels(
            target_head_indices,
            target_mask,
            validation_indices,
            len(target_metadata["target_names"]),
        )
    else:
        target_head_indices = np.zeros((row_count, 1), dtype=np.int64)
        normalized_targets = np.zeros((row_count, 1), dtype=np.float32)
        target_mask = np.zeros((row_count, 1), dtype=bool)
        target_metadata = {
            "target_names": [],
            "target_heads": [],
            "target_units": [],
            "target_mean": np.asarray([], dtype=np.float32),
            "target_std": np.asarray([], dtype=np.float32),
            "train_counts": np.asarray([], dtype=np.int64),
            "validation_counts": np.asarray([], dtype=np.int64),
            "minimum_samples_per_head": minimum_property_samples,
            "excluded_target_heads": [],
            "excluded_target_head_count": 0,
            "max_targets_per_row": 0,
            "property_notes_are_separate": True,
            "target_head_identity": ["dataset", "property", "note", "unit"],
            "ph_conditions_are_separate": True,
            "ph_conditions": [],
            "property_manifest_audit": property_manifest_audit,
        }
    context_slot_by_head = property_context_slots(target_metadata)

    split_path = LOSS_DIR / "predefined_split_indices.json"
    write_json_atomic(
        split_path,
        {
            **split_metadata,
            "training_indices": train_indices.tolist(),
            "validation_indices": validation_indices.tolist(),
        },
    )

    full_training_dataset = SpectraDataset(
        SPECTRA_PATH,
        encoded_context,
        target_head_indices,
        normalized_targets,
        target_mask,
        bandnum,
        bandwidth,
        row_count,
    )
    train_dataset = Subset(full_training_dataset, train_indices)
    validation_dataset = Subset(full_training_dataset, validation_indices)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=amp_enabled,
        persistent_workers=num_workers > 0,
        generator=training_generators["data_order"],
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batchsize,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=amp_enabled,
        persistent_workers=num_workers > 0,
    )

    aem = psm.AutoEncoder(
        10,
        context_metadata["vocab_size"],
        context_metadata["pad_index"],
        bandwidth,
        bandnum,
        maxwords=context_metadata.get("max_context_tokens", max_context_tokens),
        hidden_dim=hidden_dim,
    ).to(device)
    aem.apply(weights_init)
    property_predictor = None
    if PROPERTY_CONSTRAINT_ENABLED:
        # Predictor construction must not change PadSpectra initialization or
        # the subsequent stochastic masking stream in paired pure/constrained
        # comparisons that use the common training seed.
        rng_devices = [torch.cuda.current_device()] if amp_enabled else []
        with torch.random.fork_rng(devices=rng_devices):
            torch.manual_seed(TRAINING_SEED + 1)
            if amp_enabled:
                torch.cuda.manual_seed_all(TRAINING_SEED + 1)
            property_predictor = DatasetPropertyLinearPredictor(
                wavelength_count=bandnum * bandwidth + 1,
                target_count=len(target_metadata["target_names"]),
            ).to(device)

    ema_model = prepare_ema_model(aem)
    ema_property_predictor = (
        prepare_ema_model(property_predictor)
        if property_predictor is not None
        else None
    )
    scaler = GradScaler(device=device.type, enabled=amp_enabled)
    criterion = MAELoss()

    trainable_parameters = list(aem.parameters())
    if property_predictor is not None:
        trainable_parameters.extend(property_predictor.parameters())
    adam_kwargs = {"lr": learningRate}
    if amp_enabled:
        adam_kwargs["fused"] = True
    try:
        optimizer = optim.Adam(trainable_parameters, **adam_kwargs)
    except TypeError:
        adam_kwargs.pop("fused", None)
        optimizer = optim.Adam(trainable_parameters, **adam_kwargs)

    def lr_lambda(current_step):
        return learning_rate_factor(current_step, len(train_loader))

    scheduler = LambdaLR(optimizer, lr_lambda)

    print(
        f"Independent training variant: {TRAINING_VARIANT}; run name: "
        f"{TRAINING_RUN_NAME}; conditioning mode: "
        f"{TRAINING_CONDITIONING_MODE}; property constraint: "
        f"{PROPERTY_CONSTRAINT_ENABLED}; spectrometer dropout: "
        f"{SPECTROMETER_DROPOUT_ENABLED} "
        f"(active p={TRAINING_SPECTROMETER_DROPOUT_PROBABILITY:g}); "
        f"measurement-mode dropout: "
        f"{MEASUREMENT_MODE_DROPOUT_ENABLED} "
        f"(active p={TRAINING_MEASUREMENT_MODE_DROPOUT_PROBABILITY:g}); seed: "
        f"{TRAINING_SEED}; feed-forward: {TRAINING_FEED_FORWARD_TYPE}; "
        f"output: {RUN_DIR}"
    )
    print(
        f"Context encoding: {CONTEXT_ENCODING}; maximum "
        f"{max_context_tokens} word/value tokens per context entry; "
        f"checkpoint format: {training_checkpoint_format_version}."
    )
    plateau_description = (
        f"hold {learningRate:g} through epoch {lr_plateau_end_epoch:,}, then "
        if lr_plateau_end_epoch >= 0
        else "begin the primary cosine immediately after warm-up, then "
    )
    continuation_description = (
        f"; the continuation cosine then reaches "
        f"{continuation_final_learning_rate:g} at epoch {num_epochs:,}"
        if num_epochs > initial_schedule_end_epoch
        else "; no continuation segment"
    )
    print(
        f"Learning-rate schedule: warm up from 0 to {learningRate:g} over "
        f"{warmup_steps:,} successful optimizer steps; {plateau_description}"
        f"decay to {learningRate * 0.1:g} at epoch "
        f"{initial_schedule_end_epoch:,}{continuation_description}."
    )
    print(
        "Property manifest audit: "
        f"{property_manifest_audit['retained_property_variant_count']:,} retained "
        f"variants, each with at least "
        f"{property_manifest_audit['minimum_property_variant_count']} final values; "
        "predictor heads require only one training label because preprocessing "
        "already enforced the variant threshold."
    )
    if PROPERTY_CONSTRAINT_ENABLED:
        below_minimum_count = sum(
            head["reason"] == "optimization_count_below_explicit_test_minimum"
            for head in target_metadata["excluded_target_heads"]
        )
        no_training_count = sum(
            head["reason"] == "no_optimization_labels"
            for head in target_metadata["excluded_target_heads"]
        )
        wet_soil_excluded_count = sum(
            head["reason"] == "wet_soil_non_moisture_property"
            for head in target_metadata["excluded_target_heads"]
        )
        print(
            f"Loaded {len(target_metadata['target_names'])} eligible "
            f"dataset/property/note/unit heads; excluded {no_training_count} with no "
            f"training labels, {below_minimum_count} by an explicit test-only "
            "minimum, and "
            f"{wet_soil_excluded_count} non-moisture wet-soil heads; predictor "
            f"parameters: {sum(parameter.numel() for parameter in property_predictor.parameters()):,}"
        )
        for excluded_head in target_metadata["excluded_target_heads"]:
            print(
                f"  excluded {excluded_head['name']}: "
                f"{excluded_head['optimization_count']} optimization labels, "
                f"{excluded_head['all_training_rows_count']} total training "
                f"labels ({excluded_head['reason']})"
            )
        print(
            "All property measurement notes use separate heads; pH notes: "
            f"{', '.join(target_metadata['ph_conditions'])}"
        )
    print(
        f"Predefined indexed split: {len(train_indices):,} training rows and "
        f"{len(validation_indices):,} validation rows from the predefined "
        f"index file; {split_metadata['test_rows']:,} test rows remain unused."
    )
    print(
        f"Runtime context vocabulary: {context_metadata['vocab_size'] - 1} "
        f"tokens plus pad index {context_metadata['pad_index']}; this model "
        f"always uses {TRAINING_CONDITIONING_MODE}."
    )
    for slot_name, statistics in context_metadata["numerical_statistics"].items():
        print(
            f"  {slot_name}: {statistics['count']:,} optimization labels, "
            f"mean={statistics['mean']:.6g}, std={statistics['std']:.6g}"
        )
    for name, train_count, validation_count in zip(
        target_metadata["target_names"],
        target_metadata["train_counts"],
        target_metadata["validation_counts"],
    ):
        print(
            f"  {name}: {int(train_count):,} training labels; "
            f"{int(validation_count):,} validation labels"
        )
    if PROPERTY_CONSTRAINT_ENABLED:
        print(
            f"Starting predictor-constrained training; property loss activates "
            f"at epoch {property_start_epoch:,}, has weight "
            f"{property_loss_weight:g}, uses one SNV-linear head per "
            "dataset/property/note/unit definition, and consumes the raw "
            "PadSpectra output. Predictor spectral weights and standardized "
            "biases start at zero."
        )
    else:
        print("Starting pure padding training; no property predictor or loss exists.")

    losses = []
    spectral_losses = []
    spectral_objective_losses = []
    spectral_tv_losses = []
    spectral_negative_losses = []
    spectral_reconstruction_points = []
    property_losses = []
    validation_epochs = []
    validation_total_losses = []
    validation_spectral_losses = []
    validation_spectral_rmse = []
    validation_spectral_objectives = []
    validation_property_losses = []
    best_validation_spectral = float("inf")
    best_validation_combined = float("inf")
    global_step = 0
    start_epoch = 0
    latest_validation_metrics = None

    resume_checkpoint_value = os.environ.get(resume_checkpoint_env)
    if resume_checkpoint_value:
        resume_checkpoint_path = Path(resume_checkpoint_value)
        if not resume_checkpoint_path.is_absolute():
            resume_checkpoint_path = ROOT_DIR / resume_checkpoint_path
        resume_state = load_training_checkpoint(
            resume_checkpoint_path,
            aem,
            ema_model,
            property_predictor,
            ema_property_predictor,
            optimizer,
            scheduler,
            scaler,
            target_metadata,
            context_metadata,
            split_metadata,
            training_generators,
        )
        history = resume_state["history"]
        losses = list(history["total"])
        spectral_losses = list(history["spectral"])
        spectral_objective_losses = list(history["spectral_objective"])
        spectral_tv_losses = list(history["spectral_tv"])
        spectral_negative_losses = list(history["spectral_negative"])
        spectral_reconstruction_points = list(
            history["spectral_reconstruction_points"]
        )
        property_losses = list(history["property"])
        validation_epochs = list(history["validation_epochs"])
        validation_total_losses = list(history["validation_total"])
        validation_spectral_losses = list(history["validation_spectral"])
        validation_spectral_rmse = list(history["validation_spectral_rmse"])
        validation_spectral_objectives = list(
            history["validation_spectral_objective"]
        )
        validation_property_losses = list(history["validation_property"])
        best_validation_spectral = resume_state["best_validation_spectral"]
        best_validation_combined = resume_state["best_validation_combined"]
        global_step = resume_state["global_step"]
        start_epoch = resume_state["start_epoch"]
        latest_validation_metrics = resume_state["validation"]
        print(
            f"Resumed {resume_checkpoint_path} at epoch {start_epoch:,}, "
            f"global step {global_step:,}."
        )
        if resume_state["validation_protocol_reset"]:
            print(
                "Validation protocol changed to all available conditioning "
                "context with spectral-only checkpoint selection; resetting "
                "the best-validation baseline."
            )

    def history_payload():
        return {
            "total": losses,
            "spectral": spectral_losses,
            "spectral_objective": spectral_objective_losses,
            "spectral_tv": spectral_tv_losses,
            "spectral_negative": spectral_negative_losses,
            "spectral_reconstruction_points": spectral_reconstruction_points,
            "property": property_losses,
            "validation_epochs": validation_epochs,
            "validation_total": validation_total_losses,
            "validation_spectral": validation_spectral_losses,
            "validation_spectral_rmse": validation_spectral_rmse,
            "validation_spectral_objective": validation_spectral_objectives,
            "validation_property": validation_property_losses,
        }

    for epoch in range(start_epoch, num_epochs + 1):
        aem.train()
        if property_predictor is not None:
            property_predictor.train()
        property_loss_active = (
            PROPERTY_CONSTRAINT_ENABLED and epoch >= property_start_epoch
        )
        epoch_loss = torch.zeros((), dtype=torch.float64, device=device)
        epoch_spectral_objective = torch.zeros_like(epoch_loss)
        epoch_reconstruction_absolute_error = torch.zeros_like(epoch_loss)
        epoch_reconstruction_points = torch.zeros(
            (), dtype=torch.int64, device=device
        )
        epoch_tv_loss = torch.zeros_like(epoch_loss)
        epoch_negative_loss = torch.zeros_like(epoch_loss)
        epoch_property_loss = torch.zeros_like(epoch_loss)
        property_batches = 0

        for batch_index, batch in enumerate(train_loader):
            (
                prop_batch,
                spectra_batch,
                spectrum_2500_batch,
                property_head_index_batch,
                property_target_batch,
                property_mask_batch,
            ) = move_batch_to_device(batch)

            spectra_batch_input, non_zero_mask = build_augmented_batch(
                spectra_batch,
                epoch,
                batch_index,
                generator=training_generators["spectral_mask"],
            )

            prop_batch, _conditioning_modes = apply_conditioning_schedule(
                prop_batch,
                context_metadata["pad_index"],
                fixed_mode=TRAINING_CONDITIONING_MODE_INDEX,
                fixed_mode_spectrometer_dropout=(
                    SPECTROMETER_DROPOUT_ENABLED
                ),
                fixed_mode_measurement_dropout=(
                    MEASUREMENT_MODE_DROPOUT_ENABLED
                ),
                spectrometer_generator=training_generators[
                    "spectrometer_dropout"
                ],
                measurement_generator=training_generators[
                    "measurement_dropout"
                ],
                property_generator=training_generators["property_dropout"],
                spectrometer_dropout_probability=(
                    TRAINING_SPECTROMETER_DROPOUT_PROBABILITY
                ),
                measurement_mode_dropout_probability=(
                    TRAINING_MEASUREMENT_MODE_DROPOUT_PROBABILITY
                ),
            )

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=amp_enabled):
                padspectra, pad_mask = aem(spectra_batch_input, prop_batch)

            spectral_components = criterion.components(
                spectra_batch[non_zero_mask],
                padspectra[non_zero_mask],
                pad_mask[non_zero_mask],
            )
            spectral_objective = spectral_components["objective"]

            property_loss = spectral_objective.new_zeros(())
            with autocast(device_type=device.type, enabled=amp_enabled):
                if property_loss_active:
                    property_mask_batch = remove_conditioned_property_targets(
                        property_mask_batch,
                        property_head_index_batch,
                        prop_batch,
                        context_slot_by_head,
                        context_metadata["pad_index"],
                    )
                    usable_property_mask = (
                        property_mask_batch & non_zero_mask.unsqueeze(1)
                    )
                    property_spectra = build_property_spectra(
                        padspectra,
                        spectrum_2500_batch,
                    )
                    property_spectra = snv_transform(property_spectra)
                    (
                        property_prediction,
                        valid_property_target,
                        valid_heads,
                    ) = predict_valid_property_pairs(
                        property_predictor,
                        property_spectra,
                        property_head_index_batch,
                        property_target_batch,
                        usable_property_mask,
                    )
                    property_loss = property_pair_loss(
                        property_prediction,
                        valid_property_target.to(property_prediction.dtype),
                        valid_heads,
                        len(target_metadata["target_names"]),
                    )

            loss = spectral_objective + property_loss_weight * property_loss

            scaler.scale(loss).backward()
            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    max_grad_norm,
                )
            scale_before_step = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer_step_succeeded = (
                not scaler.is_enabled() or scaler.get_scale() >= scale_before_step
            )

            if optimizer_step_succeeded:
                completed_optimizer_steps = global_step + 1
                if should_update_ema(completed_optimizer_steps):
                    update_ema(ema_model, aem, ema_decay)
                    if property_loss_active:
                        update_ema(
                            ema_property_predictor,
                            property_predictor,
                            ema_decay,
                        )
                scheduler.step()

            epoch_loss.add_(loss.detach().to(torch.float64))
            epoch_spectral_objective.add_(
                spectral_objective.detach().to(torch.float64)
            )
            epoch_reconstruction_absolute_error.add_(
                spectral_components[
                    "reconstruction_absolute_error_sum"
                ].detach().to(torch.float64)
            )
            epoch_reconstruction_points.add_(
                spectral_components["reconstruction_points"].detach().to(torch.int64)
            )
            epoch_tv_loss.add_(
                spectral_components["tv"].detach().to(torch.float64)
            )
            epoch_negative_loss.add_(
                spectral_components["negative"].detach().to(torch.float64)
            )
            if property_loss_active:
                epoch_property_loss.add_(property_loss.detach().to(torch.float64))
                if valid_heads.numel() > 0:
                    property_batches += 1
            if optimizer_step_succeeded:
                global_step += 1

        avg_loss = (epoch_loss / len(train_loader)).item()
        reconstruction_point_count = int(epoch_reconstruction_points.item())
        if reconstruction_point_count == 0:
            raise RuntimeError("Training epoch produced zero reconstruction points")
        avg_spectral_loss = (
            epoch_reconstruction_absolute_error / epoch_reconstruction_points
        ).item()
        avg_spectral_objective = (
            epoch_spectral_objective / len(train_loader)
        ).item()
        avg_spectral_tv = (epoch_tv_loss / len(train_loader)).item()
        avg_spectral_negative = (epoch_negative_loss / len(train_loader)).item()
        avg_property_loss = (
            (epoch_property_loss / property_batches).item()
            if property_batches
            else 0.0
        )
        losses.append(avg_loss)
        spectral_losses.append(avg_spectral_loss)
        spectral_objective_losses.append(avg_spectral_objective)
        spectral_tv_losses.append(avg_spectral_tv)
        spectral_negative_losses.append(avg_spectral_negative)
        spectral_reconstruction_points.append(reconstruction_point_count)
        property_losses.append(avg_property_loss)

        if epoch % validation_interval == 0:
            validation_metrics = evaluate_validation(
                ema_model,
                ema_property_predictor,
                validation_loader,
                criterion,
                epoch,
                target_metadata,
                context_metadata,
                context_slot_by_head,
            )
            latest_validation_metrics = validation_metrics
            validation_epochs.append(epoch)
            validation_total_losses.append(validation_metrics["total"])
            validation_spectral_losses.append(validation_metrics["spectral"])
            validation_spectral_rmse.append(validation_metrics["spectral_rmse"])
            validation_spectral_objectives.append(
                validation_metrics["spectral_objective"]
            )
            validation_property_losses.append(validation_metrics["property"])
            append_json_line(
                LOSS_DIR / "validation_property_metrics.jsonl",
                {"epoch": epoch, **validation_metrics},
            )
            print(
                f"Epoch {epoch:,} | train total {avg_loss:.6e}, "
                f"reconstruction MAE {avg_spectral_loss:.6e}, "
                f"spectral objective {avg_spectral_objective:.6e}, "
                f"property {avg_property_loss:.6f} | "
                f"validation total {validation_metrics['total']:.6e}, "
                f"reconstruction MAE {validation_metrics['spectral']:.6e}, "
                f"RMSE {validation_metrics['spectral_rmse']:.6e}, "
                f"objective {validation_metrics['spectral_objective']:.6e}, "
                f"points {validation_metrics['spectral_reconstruction_points']:,}, "
                f"property {validation_metrics['property']:.6f}"
            )

            if validation_metrics["spectral"] < best_validation_spectral:
                best_validation_spectral = validation_metrics["spectral"]
                save_training_checkpoint(
                    WEIGHTS_DIR
                    / f"training_checkpoint_{hidden_dim}_best_validation.pth",
                    "best_validation",
                    aem,
                    ema_model,
                    property_predictor,
                    ema_property_predictor,
                    optimizer,
                    scheduler,
                    scaler,
                    target_metadata,
                    context_metadata,
                    split_metadata,
                    epoch,
                    global_step,
                    validation_metrics,
                    history_payload(),
                    best_validation_spectral,
                    best_validation_combined,
                    training_generators,
                )

            combined_checkpoint_eligible = (
                PROPERTY_CONSTRAINT_ENABLED
                and validation_metrics["property_active"]
            )
            if (
                combined_checkpoint_eligible
                and validation_metrics["total"] < best_validation_combined
            ):
                best_validation_combined = validation_metrics["total"]
                save_training_checkpoint(
                    WEIGHTS_DIR
                    / f"training_checkpoint_{hidden_dim}_best_combined_validation.pth",
                    "best_combined_validation",
                    aem,
                    ema_model,
                    property_predictor,
                    ema_property_predictor,
                    optimizer,
                    scheduler,
                    scaler,
                    target_metadata,
                    context_metadata,
                    split_metadata,
                    epoch,
                    global_step,
                    validation_metrics,
                    history_payload(),
                    best_validation_spectral,
                    best_validation_combined,
                    training_generators,
                )

        if epoch % checkpoint_interval == 0:
            save_training_checkpoint(
                WEIGHTS_DIR / f"training_checkpoint_{hidden_dim}_latest.pth",
                "latest",
                aem,
                ema_model,
                property_predictor,
                ema_property_predictor,
                optimizer,
                scheduler,
                scaler,
                target_metadata,
                context_metadata,
                split_metadata,
                epoch,
                global_step,
                latest_validation_metrics,
                history_payload(),
                best_validation_spectral,
                best_validation_combined,
                training_generators,
            )

        if should_save_ema_snapshot(epoch):
            torch.save(
                ema_model.state_dict(),
                WEIGHTS_DIR / f"aem_{hidden_dim}_ema_epoch_{epoch}.pth",
            )
            if property_loss_active:
                save_property_predictor(
                    ema_property_predictor,
                    target_metadata,
                    epoch,
                    global_step,
                )

        metrics_write_due = (
            epoch % metrics_flush_interval == 0
            or epoch % checkpoint_interval == 0
            or epoch == num_epochs
        )
        if metrics_write_due:
            loss_path = (
                LOSS_DIR / f"loss4AE{hidden_dim}{str(learningRate)[-3:]}.json"
            )
            write_json_atomic(loss_path, losses)

            metrics_path = (
                LOSS_DIR
                / f"training_metrics_{hidden_dim}_{str(learningRate)[-3:]}.json"
            )
            write_json_atomic(
                metrics_path,
                {
                    "total": losses,
                    "spectral": spectral_losses,
                    "spectral_objective": spectral_objective_losses,
                    "spectral_tv": spectral_tv_losses,
                    "spectral_negative": spectral_negative_losses,
                    "spectral_reconstruction_points": (
                        spectral_reconstruction_points
                    ),
                    "property": property_losses,
                    "validation_epochs": validation_epochs,
                    "validation_total": validation_total_losses,
                    "validation_spectral": validation_spectral_losses,
                    "validation_spectral_rmse": validation_spectral_rmse,
                    "validation_spectral_objective": (
                        validation_spectral_objectives
                    ),
                    "validation_property": validation_property_losses,
                    "validation_property_details": (
                        "validation_property_metrics.jsonl"
                    ),
                    "best_validation_spectral": (
                        best_validation_spectral
                        if math.isfinite(best_validation_spectral)
                        else None
                    ),
                    "best_validation_combined": (
                        best_validation_combined
                        if math.isfinite(best_validation_combined)
                        else None
                    ),
                    "best_validation_metric": best_validation_metric,
                    "secondary_validation_metric": secondary_validation_metric,
                    "validation_conditioning_policy": (
                        validation_conditioning_policy
                    ),
                    "validation_interval": validation_interval,
                    "validation_mask_seed": validation_mask_seed,
                    "validation_mask_epoch": validation_mask_epoch,
                    "validation_weights": "ema",
                    "ema_decay": ema_decay,
                    "ema_update_interval": ema_update_interval,
                    "minimum_observed_bands": minimum_observed_bands,
                    "maximum_missing_bands": maximum_missing_bands,
                    "checkpoint_interval": checkpoint_interval,
                    "ema_checkpoint_interval": ema_checkpoint_interval,
                    "periodic_ema_snapshots_enabled": True,
                    "metrics_flush_interval": metrics_flush_interval,
                    "learning_rate_schedule": (
                        "warmup_plateau_two_stage_cosine_v1"
                    ),
                    "warmup_steps": warmup_steps,
                    "lr_plateau_end_epoch": lr_plateau_end_epoch,
                    "training_end_epoch": num_epochs,
                    "initial_schedule_end_epoch": initial_schedule_end_epoch,
                    "continuation_final_learning_rate": (
                        continuation_final_learning_rate
                    ),
                    "training_variant": TRAINING_VARIANT,
                    "training_run_name": TRAINING_RUN_NAME,
                    "conditioning_mode": TRAINING_CONDITIONING_MODE,
                    "property_constraint_enabled": PROPERTY_CONSTRAINT_ENABLED,
                    "spectrometer_dropout_enabled": (
                        SPECTROMETER_DROPOUT_ENABLED
                    ),
                    "spectrometer_dropout_probability": (
                        TRAINING_SPECTROMETER_DROPOUT_PROBABILITY
                    ),
                    "measurement_mode_dropout_enabled": (
                        MEASUREMENT_MODE_DROPOUT_ENABLED
                    ),
                    "measurement_mode_dropout_probability": (
                        TRAINING_MEASUREMENT_MODE_DROPOUT_PROBABILITY
                    ),
                    "feed_forward_type": TRAINING_FEED_FORWARD_TYPE,
                    "training_seed": TRAINING_SEED,
                    "training_generator_seeds": TRAINING_GENERATOR_SEEDS,
                    "run_directory": str(RUN_DIR),
                    "split": split_metadata,
                    "property_start_epoch": property_start_epoch,
                    "property_input_spectra": property_input_spectra,
                    "property_predictor_type": property_predictor_type,
                    "property_predictor_initialization": (
                        "zero_standardized_target_mean"
                    ),
                    "property_snv_epsilon": property_snv_epsilon,
                    "property_loss_weight": property_loss_weight,
                    "minimum_property_samples": minimum_property_samples,
                    "property_manifest_audit": property_manifest_audit,
                    "excluded_target_heads": target_metadata[
                        "excluded_target_heads"
                    ],
                    "wet_soil_constraint": target_metadata.get(
                        "wet_soil_constraint"
                    ),
                    "context": context_metadata,
                    "property_notes_are_separate": True,
                    "target_head_identity": target_metadata[
                        "target_head_identity"
                    ],
                    "ph_conditions_are_separate": True,
                    "ph_conditions": target_metadata["ph_conditions"],
                    "target_heads": target_metadata["target_heads"],
                    "target_names": target_metadata["target_names"],
                    "target_units": target_metadata["target_units"],
                    "training_target_counts": (
                        target_metadata["train_counts"].tolist()
                    ),
                    "validation_target_counts": (
                        target_metadata["validation_counts"].tolist()
                    ),
                },
            )

if __name__ == "__main__":
    main()
