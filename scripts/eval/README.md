# `eval`

Evaluation notebook (run in google colab) for auditing the face-name handoff package (produced by [`training`](../training/README.md)) and computing paired bootstrap statistics.

## Workflow

1. **Validate inputs**: Clones the repository, verifies all handoff files exist, and checks `h5_index` alignment and unit L2 normalization across validation and test splits.
2. **Observed accuracy**: Computes `blocked_triplet_accuracy` (from [`face_name_metrics`](../training/face_name_metrics/README.md)) for `baseline_iresnet100`, `triplet_margin`, `arcface`, and `supcon`.
3. **GPU paired bootstrap (2,000 replicates)**: Derives deterministic split seeds from `root_seed=4002`, generates shared [`stratified_bootstrap_multiplicities`](../training/face_name_metrics/README.md), verifies a PyTorch CUDA `gpu_batched_triplet_accuracy` implementation against CPU `weighted_triplet_accuracy`, and runs all 8 model/split combinations.
4. **Integrity audit**: Verifies checksums from `handoff_manifest.json` across the 6 uploaded artifacts and confirms all 3 selected trials match `trial_results.csv`.
5. **Export results**: Computes paired similarity lifts and 95% percentile CIs (in percentage points) and saves final evaluation files.

## Required Inputs

* [`face_name_metrics`](../training/face_name_metrics/README.md) package
* `data/processed/manifest.csv` produced by [`preprocess`](../preprocess/README.md)
* [`output/artifacts/`](../../output/artifacts/README.md) (`handoff_manifest.json`, `trial_results.csv`, and all four `.npz` embedding files)

## Outputs

* [`final_metrics.csv`](../../output/final_metrics.csv) — Final 8-row summary (`model`, `split`, `triplet_accuracy`, `baseline_lift_pp`, `lift_ci_95`).