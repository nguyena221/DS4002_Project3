# `notebooks`

Main training pipeline executed on [looseleaf](https://looseleafcloud.org/) notebooks for loading pretrained weights, validating the environment, training the 10-trial metric-learning grid, and exporting the handoff package.

## Execution Order

| Step | Notebook Stage | Purpose | Key Outputs |
| :--- | :--- | :--- | :--- |
| 1 | **LooseLeaf & Pretrained Weight Setup** | Unzips pretrained IResNet100 (`model-r100-ii.zip`), converts to `.pth` via `pytorch-insightface`, and prints its hash. | `models/pretrained/iresnet100.pth` not commited for size reasons |
| 2 | **Setup & IResNet-100 Smoke Test** | Verifies pinned packages with versions (`torch==2.8.0`, `pytorch-metric-learning==2.8.1`, `h5py==3.16.0`), validates `manifest.csv`/`faces.h5` (produced by [preprocess](../../preprocess/README.md)), checks weight hash, and runs a 1 image pass to make sure things are working. | `artifacts/face_name_metric_learning/setup_environment.json` relocated to [output/artifacts/](../../../output/artifacts/setup_environment.json)|
| 3 | **Export Pretrained Baseline Embeddings** | Runs `iresnet100.pth` in eval mode over validation/test splits to produce baseline embeddings. | `artifacts/face_name_metric_learning/embeddings/baseline_iresnet100.npz` relocated to [output/artifacts/embeddings](../../../output/artifacts/embeddings/baseline_iresnet100.npz) |
| 4 | **Candidate Fitting & Model Selection** | Runs 10 trials over three loss functions with various hyperparameters (`3` Triplet Margin, `4` ArcFace, `3` SupCon) and marks one winner per loss function based on validation data performance. | `artifacts/face_name_metric_learning/trial_results.csv` relocated to [output/artifacts/](../../../output/artifacts/trial_results.csv), `models/metric_learning/<trial_id>/best.pt` not committed for size reasons |
| 5 | **Export Selected Embeddings & Manifest** | Exports validation/test embeddings for the 3 selected checkpoints, and builds metadata for clean handoff to the [scripts/eval](../../eval/README.md) stage. | `triplet_margin_selected.npz`, `arcface_selected.npz`, `supcon_selected.npz`, all located in [output/artifacts/embeddings](../../../output/artifacts/embeddings/) and `handoff_manifest.json` relocated to [output/artifacts](../../../output/artifacts/handoff_manifest.json) |

## Required Inputs

* `data/processed/manifest.csv` and `data/processed/faces.h5` (100 classes: 45,078 train / 9,660 validation / 9,660 test) produced by [preprocess](../../preprocess/README.md).
* `model-r100-ii.zip` (InsightFace MS1MV2 R100 archive) for Step 1.
* CUDA GPU runtime, and lots of time to wait for fine tuning to finish