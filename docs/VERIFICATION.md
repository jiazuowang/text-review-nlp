# Public-release verification

Verified on 16 November 2026 with Windows and Python 3.14.6, using an existing environment matching the model dependencies.

Command: `python -m scripts.test_public`

- 68 tests discovered: 67 passed, 1 explicitly skipped.
- The skipped test requires the excluded original frozen training/validation dataset and manifests.
- All 16 application tests passed, including three-model inference, both threshold modes, invalid input, cancellation, stale GUI result handling, model hash verification, and PDF/DOCX round trips.
- The POS resource check and feature/model serialization checks passed.
- Training, feature, editing and evaluation logic were checked with synthetic test fixtures. This is not a fresh reproduction of the corpus-level experiments.

The three model hashes match the original implementation package and final group delivery. The public edition retains the final group's test cleanup change. Core application and model-training algorithms were not modified for publication.

The private dataset, course documents, original per-document results, virtual environments, and machine-specific run records are excluded. Privacy checks searched the selected text files and decompressed model artifacts for known personal identifiers, private machine paths, and common credential patterns. The publication includes the runtime model/POS/font files and their notices.

No fresh dependency installation, macOS/Linux run, or full-dataset retraining is claimed by this record.
