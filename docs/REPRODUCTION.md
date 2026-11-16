# Reproduction scope

## Included and runnable

The desktop app, CLI, three trained models, POS resource, font fallbacks, synthetic demo text and public test runner are included. Inference runs locally after dependency installation. Use the commands in the main README.

## Original experiments

Source for dataset preparation, model training, editing analysis and evaluation is retained. Exact experiment reproduction additionally needs the original frozen data partitions, `CHECKSUMS.json`, relevant experiment artifacts, and validation/test fixtures. These are excluded from the public edition.

The original `verify.py`, `verify_submission.py`, `scripts/team_reproduce.py`, and `scripts/verify_module_f.py` target the complete private course package. They are retained as implementation references, not presented as runnable public-release entry points. Use `python -m scripts.test_public` for this edition.

Dataset preparation is in `src/data/prepare.py`; training entry points are `src/models/tfidf_lr.py`, `train_style.py`, and `extension.py`. The training loader validates frozen split membership and hashes. Newly generated datasets or models are not silently substituted into the archived downstream stages.

No claim is made that a fresh retraining run has been completed as part of this publication. Recorded aggregate metrics are preserved separately from public-release software verification.
