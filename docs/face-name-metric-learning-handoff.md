# LooseLeaf Training and Colab Handoff

This document is the execution contract between the LooseLeaf training stage and
the Google Colab evaluation stage. It describes generated artifacts and
notebook responsibilities; it does not replace executable notebook code.

## Current data contract

Use the prepared data already committed to the repository:

- `data/processed/faces.h5` is local-only and contains 64,398 aligned RGB faces
  with shape `(3, 112, 112)`, normalized to `[-1, 1]`.
- `data/processed/manifest.csv` is committed and is the shared metadata source.
- `h5_index` is the stable identity key across notebooks, embeddings, and
  evaluation results.
- The split contains 45,078 training, 9,660 validation, and 9,660 test images
  across 100 names.
- `first_name` and `label` define the class. A positive has the same
  `first_name` as the anchor and a different original `h5_index`; a negative
  has a different `first_name`.

Do not use the earlier 30,000-image/75-name estimate in implementation. Derive
counts from the manifest at runtime and assert them against these expected
counts so stale data fails visibly.

## Training design

The stage trains 10 candidate fits: one fit per candidate configuration and
seed. The root seed is `4002`. Use child RNG streams for batch sampling,
augmentation, loss-level sampling, and bootstrap generation so these operations
are independently reproducible.

| Loss | Candidate configurations | Candidates |
|---|---|---:|
| Triplet Margin | margin: `0.1`, `0.2`, `0.4` | 3 |
| ArcFace | margin: `0.2`, `0.4`; scale: `30`, `64` | 4 |
| SupCon | temperature: `0.07`, `0.10`, `0.20` | 3 |

Shared settings:

- Pretrained IResNet-100 facial-recognition backbone, 512-dimensional output.
- Adam learning rate `1e-5`, betas `(0.9, 0.999)`, and weight decay `0`.
- All backbone parameters are trainable.
- CUDA automatic mixed precision for training; exported embeddings are float32.
- Maximum 15 epochs.
- Select the best epoch by exhaustive validation triplet accuracy.
- Early stop after 4 epochs without an improvement of at least `1e-4`.
- Name-balanced batches of 16 names by 8 images (`P x K`, batch size 128),
  falling back to 8 names by 8 images if required by GPU memory.
- Training augmentation: horizontal flip with probability 0.5. Do not augment
  validation or test embeddings.
- One stochastic fit per candidate configuration. Do not repeat a configuration
  with additional training seeds.

A candidate fit is one complete training trajectory for one loss/configuration
and seed. Fitting the identical configuration again with a different seed
would be an additional candidate fit; such fits are out of scope.

### Loss-specific sampling

Each training batch loads `P` names and `K` different images per name. For an
anchor, Triplet Margin chooses a positive from the other `K - 1` images of its
name and a negative uniformly from the `(P - 1) * K` images of other names in
the batch. Do not add hard-negative or semi-hard ranking.

ArcFace trains a temporary 100-class classifier head alongside the trainable
backbone. The head is not part of the evaluation handoff because Colab only
needs backbone embeddings. SupCon treats same-name examples in the batch as
positives according to the library loss implementation and should record the
exact library version and sampler behavior.

After the next batch is assembled, it gets one optimizer step. Batch construction
continues until the sampler has approximately passed over the 45,078 training
images for that epoch. Record actual `steps_per_epoch` and `images_seen`; do not
silently claim every image was observed if the final incomplete group is
dropped.

## Model selection and evaluation boundary

LooseLeaf owns candidate training, validation-based selection, and embedding
export. Colab owns final metric reporting and paired bootstrap analysis.

For each epoch, embed all 9,660 validation images once. Compute
`cos(anchor, positive)` and `cos(anchor, negative)` for every valid ordered
triplet and persist only the scalar triplet accuracy. Use blocked matrix
operations; never materialize roughly 8.9 billion Python triplet objects.
Ties count as incorrect because the criterion is strict `>`.

Select one best-epoch fit for each loss. Persist the selected row by
`trial_id`; do not retrain the selected configuration on train plus validation.
The evaluation package therefore contains one pretrained baseline and three
selected fine-tuned model embedding sets.

Use deterministic tie breaks:

1. Highest validation triplet accuracy.
2. Smallest margin/temperature.
3. For ArcFace ties, smaller scale.

## Handoff directory

Create this committed directory:

```text
artifacts/face_name_metric_learning/
|-- README.md
|-- handoff_manifest.json
|-- trial_results.csv
`-- embeddings/
    |-- baseline_iresnet100.npz
    |-- triplet_margin_selected.npz
    |-- arcface_selected.npz
    `-- supcon_selected.npz
```

The package is intentionally not a training checkpoint archive. The four NPZ
files may be tracked directly if each remains below GitHub's per-file limit.
Candidate checkpoints remain under local ignored `models/metric_learning/` and
are not teammate requirements.

### `README.md`

The teammate-facing README should let an evaluator start in under five minutes.
It should contain:

1. The exact files to upload to Colab.
2. A short NumPy load/shape/dtype example.
3. The `h5_index` join back to the committed manifest.
4. Metric definitions and strict tie behavior.
5. Expected split/class shapes and integrity assertions.
6. The selected trial IDs for each loss.
7. The 2,000-replicate bootstrap seed/specification.
8. The expected output table and plots.

Keep training details in this specification, notebooks, and manifests.

### Embedding files

Each NPZ file contains:

| Key | Type/shape | Meaning |
|---|---|---|
| `validation_h5_index` | `int64`, `(9660,)` | Original image IDs, in embedding-row order |
| `validation_embeddings` | `float32`, `(9660, 512)` | Unit-normalized validation vectors |
| `test_h5_index` | `int64`, `(9660,)` | Original image IDs, in embedding-row order |
| `test_embeddings` | `float32`, `(9660, 512)` | Unit-normalized test vectors |

Required invariants:

- Index arrays have no duplicates, are members of the corresponding manifest
  split, and exactly cover that split.
- Embedding rows contain only finite values.
- Every row has an L2 norm within a documented tolerance of 1.
- The validation and test index orders may differ between files, so evaluation
  code must align by `h5_index`, not row position.

The manifest contains filename, first name, label, and split; do not duplicate
that metadata in each NPZ. No original face images are required by Colab.

### `trial_results.csv`

Write one row for each candidate fit:

- `trial_id`
- `loss`
- `hyperparameters_json`
- `seed`
- `epochs_attempted`
- `best_epoch`
- `best_validation_triplet_accuracy`
- `early_stopped`
- `images_seen`
- `steps_per_epoch`
- `device`
- `selected`
- `notes`

Use deterministic IDs such as:

- `triplet_margin0p2_seed4002`
- `arcface_m0p2_s64_seed4002`
- `supcon_t0p07_seed4002`

The selected baseline is not a candidate fit. List it in the handoff manifest.

### `handoff_manifest.json`

Use `schema_version: 1` with stable model filenames. Record:

- Generation time in ISO UTC and Git commit hash when available.
- SHA-256 checksums for `faces.h5`, `manifest.csv`, trial table, and each NPZ.
- Exact Python, PyTorch, CUDA, pytorch-metric-learning, and h5py versions,
  plus the pretrained weight identifier/checksum.
- Root seed, child-seed policy, split counts, class count, and metric definitions.
- Device used for candidate training.
- Selected `trial_id` for Triplet, ArcFace, and SupCon.
- NPZ shape, dtype, index coverage, and norm/finite checks.
- The audit timestamp and `pass`, `fail`, or `not_run` result. A generated
  manifest starts as `not_run`; only the independent audit may record `pass`.

Production execution status: this repository currently contains the executable
pipeline and a README template, but not production trial rows, a production
manifest, or production embedding files. Generate and independently audit
those files before treating the handoff as accepted.

Do not add timestamps to the stable embedding filenames. Package history is
tracked by Git and checksums.

## Notebook set

Commit four rerunnable notebooks. They must have cleared outputs, a single
configuration cell, explicit paths, and fail-fast assertions.

### `notebooks/00_setup_and_smoke_test.ipynb`

- Install/import pinned runtime packages.
- Record Python, package, GPU, CUDA, and device information.
- Verify repository paths, manifest/HDF5 counts, class and split coverage, and
  the `h5_index` mapping.
- Load the pretrained IResNet-100 and run one forward pass on a known image.
- Assert one normalized embedding has shape `(512,)`.
- Write the environment/configuration section later consumed by the handoff
  manifest.

### `notebooks/01_export_baseline_embeddings.ipynb`

- Load the pretrained IResNet-100 in evaluation mode.
- Embed validation and test sets without gradients or augmentation.
- L2-normalize in float32.
- Write `baseline_iresnet100.npz` with the required index and embedding arrays.
- Validate index coverage, shapes, finiteness, and unit norms.

### `notebooks/02_train_metric_models.ipynb`

- Define the three candidate grids and the shared training configuration.
- Create name-balanced batches and exactly one candidate fit per configuration.
- Use the configured loss-specific sampler/head behavior.
- At each epoch, compute exhaustive validation triplet accuracy in blocks and
  retain only the best-epoch checkpoint.
- Apply early stopping and deterministic tie rules.
- Append one row to `trial_results.csv` per candidate.
- Save checkpoints only below ignored `models/metric_learning/`.
- Finish with a readable table marking the three selected trials.

### `notebooks/03_export_selected_embeddings.ipynb`

- Read and assert the three selected rows in `trial_results.csv`.
- Load each selected backbone checkpoint and export validation/test embeddings
  under the stable selected-model filenames.
- Write `handoff_manifest.json` and the teammate-facing README.
- Never change split membership or index ordering assumptions.

### Local-only `notebooks/04_audit_handoff.ipynb`

Run this before handoff, but do not commit it; `.gitignore` excludes it. It
should independently:

- Reload every NPZ.
- Join every index to `manifest.csv`.
- Verify shapes, dtypes, split coverage, uniqueness, finite values, and norms.
- Recalculate checksums and compare them to the handoff manifest.
- Verify the selected trial IDs and trial-table counts.
- Recompute at least one known validation metric per model or a deterministic
  check sample.
- Write `audit` status and timestamp into `handoff_manifest.json`.

Do not depend on LooseLeaf retaining variables for a week. Every stage must be
rerunnable from persistent files and explicit inputs.

## Colab evaluation contract

The evaluator should use:

- `data/processed/manifest.csv`
- all four selected-model NPZ files
- `trial_results.csv` and `handoff_manifest.json`

For each model and split, align embeddings to the manifest by `h5_index`, then
compute one pairwise cosine similarity matrix. The matrix can be blocked if
Colab memory is limited. Exhaustive valid triplet accuracy is calculated
without enumerating or storing the full triplet Cartesian product.

Definitions:

- `triplet_accuracy = correct_predictions / total_valid_comparisons`
- `similarity_lift = fine_tuned_triplet_accuracy - baseline_triplet_accuracy`

For uncertainty, run 2,000 paired, stratified image-level bootstrap replicates
using the manifest's root-seed/child-seed policy, deriving one stream per
`<split>` under the `bootstrap` concern. The same sampled original
images and multiplicities must be applied to all four models for each
comparison. A comparison
still requires distinct original `h5_index` values, so duplicated bootstrap
draws of one original image cannot be paired with each other. Use weighted
counts from the similarity matrix rather than rebuilding/recomputing all
triplets 2,000 times.

For each fine-tuned model, record the confidence interval of its lift against
baseline. The project treats an improvement as statistically significant when
the 95% interval excludes zero. Final outputs are one validation/test result
table, model-to-baseline lift intervals, and any agreed report plots.
