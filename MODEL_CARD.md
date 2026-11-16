# Text Review models

## Included artifacts

| Artifact | Description |
| --- | --- |
| `resources/detectors/B/pipeline.joblib` | Word TF-IDF and logistic regression |
| `resources/detectors/C/pipeline.joblib` | 32 style features and a 300-tree random forest |
| `artifacts/extension_x/pipeline.joblib` | Word/character/style feature union and logistic regression |

Model paths, SHA-256 values and operating thresholds are stored in `configs/integration_models.json`. The files are identical to the final group package. Feature implementations remain in `src/features/`; the style models also use the bundled POS resource.

The models were trained on the original-document partition of Beemo revision `cca3b817eb6d0bf7a60f885f3022696553eabb63`. Labels describe dataset generation history. Original texts, edited texts, per-document predictions and review annotations are not redistributed in this release.

## Interpretation

- Intended use: local research demonstrations and comparison of NLP approaches.
- The observed false-positive rates are substantial at the default threshold. Predictions do not prove authorship and should not determine academic misconduct or employment decisions.
- Model scores have not been independently calibrated as real-world probabilities.
- Performance on new topics, generators, languages and document lengths may differ.
- The combined model's 77.47% accuracy is a follow-up estimate on a previously inspected test partition.
- Passage highlights are sensitivity measurements, not a separately validated authorship detector.

## Loading and provenance

Joblib uses pickle-based serialization. Load only trusted artifacts; hashes detect changes but do not make arbitrary model files safe. This application checks the published file hashes before loading. The release review checked the bundled compressed artifacts for known team identifiers, personal machine paths and common credential patterns, and compared their hashes to the original implementation package. Such checks are not a formal privacy guarantee.

No runtime user documents or credentials are required to use the included models. The lexical vocabulary is learned from the public training corpus and is part of the model artifact. See `THIRD_PARTY_NOTICES.md` for upstream data conditions; this repository does not represent the models as cleared for unrestricted commercial use.
