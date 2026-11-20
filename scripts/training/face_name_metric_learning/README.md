# `face_name_metric_learning`

Python utilities for training, packaging, auditing, and evaluation for IResNet-100 name metric learning.

## Workflow

1. **Validate data** (`handoff.py`): `validate_manifest_hdf5` checks `manifest.csv` against `faces.h5` (files produced by [preprocess](../../preprocess/README.md)).
2. **Baseline** (`training.py`, `iresnet.py`): `collect_environment_record`, `load_pretrained_iresnet100`, `export_baseline_embeddings`.
3. **Train & select** (`pytorch_training.py`): `train_and_select_candidates` runs the 10-trial CUDA/AMP grid (`triplet_margin`, `arcface`, `supcon`) and selects one winner per loss in [`trial_results.csv`](../../../output/artifacts/trial_results.csv).
4. **Package** (`training.py`, `handoff.py`): `export_selected_embeddings` + `build_handoff_manifest_from_environment`.
5. **Audit** (`audit.py`): `audit_handoff_package` runs weights-free checksum and metric checks, updating `handoff_manifest.json`.
6. **Evaluate** (`acceptance.py`): `handoff_package_readiness` + `evaluate_handoff_package` (triplet accuracy, similarity lift, paired bootstrap CIs).

## Modules

| File | Purpose |
| :--- | :--- |
| `__init__.py` | Public API exports. |
| `handoff.py` | Validation, SHA-256 checksums, `h5_index` alignment, and manifest/NPZ I/O. |
| `iresnet.py` | IResNet-100 backbone and strict local weight loader. |
| `training.py` | Config, 10-trial grid, SHA-256 child seeding, P×K batching, ledger I/O, and embedding export. |
| `pytorch_training.py` | CUDA/AMP training loop, RAM image caching, OOM batch probing, and resumable execution. |
| `audit.py` | Weights-free package verification and pass/fail recording. |
| `acceptance.py` | Fresh-environment readiness and bootstrap evaluation. |

## Output

These utilities help produce a clean source of artifacts used for final evaluation (now located in the [output](../../../output/README.md) directory) including metadata (handoff_manifest.json), training and validation results for each fine tuned model (trial_results.csv), and the validation and test embeddings needed to compute final simularity lift and statistical significance tests. Utilities for evaluation can be found in [face_name_metrics](../face_name_metrics/README.md).

```text
artifacts/
├── handoff_manifest.json
├── trial_results.csv
└── embeddings/
    ├── baseline_iresnet100.npz
    ├── triplet_margin_selected.npz
    ├── arcface_selected.npz
    └── supcon_selected.npz
```