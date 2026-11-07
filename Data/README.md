# Data

This directory contains a small, de-identified Gemini example and documents
the external files required for full PadSpectra training.

## Included Gemini example

`examples/gemini_leaf_spectra_sample.xlsx` contains eight measured leaf
spectra: one reflectance and one transmittance record for each of four species
represented in the source workbook. The spectra span 350–2500 nm at 1 nm
intervals.

The workbook has two sheets:

- `README`: provenance, selection method, source checksum, and use limits
- `Spectra`: synthetic sample IDs, species, spectral type, and spectral values

Original dates, plot numbers, sample IDs, notes, and instrument IDs were
removed. This example is provided for input inspection and demos; it is not
the complete training dataset and does not reproduce the predefined data
partitions. The source checksum recorded in the workbook allows the extraction
to be traced to the maintainers' `GEMINIall.xlsx` source file.

The repository's Apache License 2.0 covers source code. Data reuse remains
subject to the terms of the original Gemini dataset.

## Full dataset layout

PadSpectra does not redistribute the compiled training dataset. Supply the
following files in this directory, or set `PADSPECTRA_DATA_DIR` to an equivalent
external directory:

| File | Description |
| --- | --- |
| `spectra.h5` | Aligned spectra with wavelengths from 350 to 2500 nm and any required masks |
| `metadata.json` | Row-aligned sample metadata and property measurements |
| `target_schema.json` | Canonical property, unit, note, and dataset target definitions |
| `split_indices.json` | Predefined train, validation, and independent test indices plus audits |
| `processing_manifest.json` | Frozen preprocessing profile and retained-property audit |
| `SHA256SUMS` | Checksums for immutable dataset artifacts |

Training verifies alignment, partition integrity, target definitions, and
checksums before optimization. Do not add restricted source datasets to this
repository.
