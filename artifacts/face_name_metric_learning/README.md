# Face-name metric-learning handoff

This document is designed to help with final evaluation. The goal of this step is to take the input files listed below and produce one csv file containing the final accuracy, difference from the baseline and a statistical confidence interval inidcating if any of our selected fine-tuned models show improvements over the baseline un-fine-tuned IResNet-100. Files used include:

- `data/processed/manifest.csv`

File listing the image names, h5 index (a unique number per image), first names, label (from 0-99 corresponding to each name), estimated age, det score, and sha hash for each of the ~64,000 images that survived pre-processing.

- `embeddings/*.npz`

These files contain all of the embeddings for each image in the validation and test data. Each file corresponds to one of the 4 models we trained (baseline un-fine-tuned iresnet100, fine-tuned triplet margin loss, fine-tuned arcface loss, and fine-tuned supcon loss). Validation over 10 total hyperparameters has already been done and the selected models were used to create the embeddings currently stored here.

Here is the full shape of each `embeddings/*.npz` file:

| Key | Type/shape | Meaning |
|---|---|---|
| `validation_h5_index` | `int64`, `(9660,)` | Original image IDs, in embedding-row order |
| `validation_embeddings` | `float32`, `(9660, 512)` | Unit-normalized validation vectors |
| `test_h5_index` | `int64`, `(9660,)` | Original image IDs, in embedding-row order |
| `test_embeddings` | `float32`, `(9660, 512)` | Unit-normalized test vectors |

- `trial_results.csv`

This file contains the results of each fine-tuning run for all 10 models selected over during validation. For all 10 trails this includes a trial_id, the best validation loss, the hyperparameters used, and other less relevant information.

- `handoff_manifest.json`

This file contains lots of metadata about the data used/produced in training. Most importantly, under "models" the file includes all of the trial_id's (the same trial_id found in `trial_results.csv`) of the selected model (for each loss function).

- `face_name_metrics/**`

This is a python package that contains `bootstrap.py` and `metrics.py`, both utilities that will aid in performing the evaluation (your LLM will know how to use them).

## Upload

Upload `data/processed/manifest.csv`, the entire `face_name_metrics/` package
directory, and this directory's:

- `trial_results.csv`
- `handoff_manifest.json`
- `embeddings/baseline_iresnet100.npz`
- `embeddings/triplet_margin_selected.npz`
- `embeddings/arcface_selected.npz`
- `embeddings/supcon_selected.npz`
- `README.md` (for reference)

In Colab, add the uploaded repository root to the import path:

```python
import sys
from pathlib import Path

root = Path(".")
sys.path.insert(0, str(root))

from face_name_metrics.bootstrap import (
    paired_bootstrap_statistics,
    stratified_bootstrap_multiplicities,
)
from face_name_metrics.metrics import blocked_triplet_accuracy
```

## Load and join

```python
import json
from pathlib import Path

import numpy as np

root = Path(".")
package = root / "artifacts" / "face_name_metric_learning"
handoff = json.loads((package / "handoff_manifest.json").read_text())
with np.load(
    package / "embeddings" / "baseline_iresnet100.npz", allow_pickle=False
) as data:
    validation_h5_index = data["validation_h5_index"]
    validation_embeddings = data["validation_embeddings"]

assert validation_h5_index.dtype == np.int64
assert validation_embeddings.dtype == np.float32
assert validation_embeddings.shape == (9_660, 512)

import csv
with (root / "data" / "processed" / "manifest.csv").open(
    newline="", encoding="utf-8"
) as stream:
    identity = {int(row["h5_index"]): row for row in csv.DictReader(stream)}

validation_metadata = [identity[int(index)] for index in validation_h5_index]
```

Always align model arrays to the target split by `h5_index`, not row position.
The exact selected IDs are `handoff["selected_trials"]`; retain that mapping
when reporting Triplet Margin, ArcFace, and SupCon.

## Integrity and metric contract

Assert 45,078 train, 9,660 validation, and 9,660 test rows across 100 names.
For every model and each evaluation split, require:

- unique `int64` indices covering that manifest split exactly once;
- `float32` finite embeddings of shape `(split_rows, 512)`;
- row L2 norms within `1e-5` of 1; and
- every SHA-256 value for an uploaded file in `handoff_manifest.json` matching
  its copied file.

The recorded `faces.h5` checksum is a source-data fingerprint and is not
verifiable in Colab because the file is intentionally not uploaded. All copied
package/shared-file checksums are still asserted directly.

Triplet accuracy is `correct / total_valid_comparisons`: a positive has the
same `first_name` and a different `h5_index`; a negative has a different
`first_name`. Count a comparison correct only when the cosine to the positive
is **strictly greater** than the cosine to the negative (ties are incorrect).
Similarity lift is fine-tuned triplet accuracy minus baseline triplet
accuracy, reported in percentage points.

## Paired bootstrap and outputs

Use the supplied `face_name_metrics.bootstrap` and `face_name_metrics.metrics`
modules. Derive the child seed exactly as follows:

```python
import hashlib
import json

def bootstrap_seed(root_seed, split):
    payload = json.dumps(
        {
            "root_seed": int(root_seed),
            "concern": "bootstrap",
            "scope": split,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")
```

For each split, load and align all four models by `h5_index`, then use one set of
shared multiplicities for all paired baseline comparisons:

```python
root_seed = int(handoff["root_seed"])
model_files = {
    "baseline_iresnet100": "baseline_iresnet100.npz",
    "triplet_margin_selected": "triplet_margin_selected.npz",
    "arcface_selected": "arcface_selected.npz",
    "supcon_selected": "supcon_selected.npz",
}

split = "validation"  # repeat for "test"
target_rows = [row for row in identity.values() if row["split"] == split]
target_h5_index = np.array([row["h5_index"] for row in target_rows], dtype=np.int64)
first_names = np.array([row["first_name"] for row in target_rows], dtype=object)

aligned = {}
for model, filename in model_files.items():
    with np.load(package / "embeddings" / filename, allow_pickle=False) as data:
        source_index = data[f"{split}_h5_index"]
        source_embeddings = data[f"{split}_embeddings"]
    positions = {int(value): position for position, value in enumerate(source_index)}
    aligned[model] = source_embeddings[
        [positions[int(value)] for value in target_h5_index]
    ]

accuracies = {
    model: blocked_triplet_accuracy(
        embeddings, target_h5_index, first_names, block_size=1024
    )
    for model, embeddings in aligned.items()
}

draw_weights = stratified_bootstrap_multiplicities(
    first_names,
    replicates=2_000,
    seed=bootstrap_seed(root_seed, split),
)

bootstrap = {}
for model in model_files:
    if model == "baseline_iresnet100":
        continue
    bootstrap[model] = paired_bootstrap_statistics(
        aligned["baseline_iresnet100"],
        aligned[model],
        target_h5_index,
        first_names,
        replicates=2_000,
        draw_weights=draw_weights,
    )
```

`paired_bootstrap_statistics` excludes comparisons between duplicate draws of
the same original `h5_index` and reuses the weighted similarity matrices. Its
`similarity_lifts` and `confidence_interval` values are decimal fractions;
multiply by 100 when reporting percentage points. Improvement is significant
when the 95% interval excludes zero.

Expected outputs are:

| model | split | triplet_accuracy | baseline_lift_pp | lift_ci_95 |
|---|---|---:|---:|---|

Produce one validation/test result table, model-to-baseline lift intervals,
and agreed report plots. The baseline lift is 0 for the baseline row.
