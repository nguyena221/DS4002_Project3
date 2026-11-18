# Face-name metric-learning handoff

> Status: **generation pending**. This repository currently contains the
> handoff contract and executable pipeline, but production candidate fitting,
> embedding export, and the independent audit have not been executed. The files
> listed below must be generated before upload; do not treat this README as an
> acceptance result.

The generated handoff is designed to contain embeddings only: no face images,
checkpoints, or model weights are required by the evaluator.

## Upload

After production generation and a successful independent audit, upload
`data/processed/manifest.csv` and this directory's:

- `trial_results.csv`
- `handoff_manifest.json`
- `embeddings/baseline_iresnet100.npz`
- `embeddings/triplet_margin_selected.npz`
- `embeddings/arcface_selected.npz`
- `embeddings/supcon_selected.npz`

Keep the `README.md` beside them for reference. Do **not** upload candidate
checkpoints or `data/processed/faces.h5`.

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

Use root seed 4002 and derive one named child seed per split with concern
`bootstrap` and scope `<split>` through the policy in the manifest to generate 2,000
paired, name-stratified, image-level bootstrap replicates. Apply each sample's
multiplicities to all four models. Exclude comparisons between
duplicate draws of the same original `h5_index`, and use weighted precomputed
similarity matrices rather than rebuilding triplets. Report a 95% interval for
each fine-tuned model's lift; improvement is significant when it excludes zero.

Expected outputs are:

| model | split | triplet_accuracy | baseline_lift_pp | lift_ci_95 |
|---|---|---:|---:|---|

Produce one validation/test result table, model-to-baseline lift intervals,
and agreed report plots. The baseline lift is 0 for the baseline row.
