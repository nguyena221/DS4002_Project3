# Export selected models and handoff manifest

Status: resolved
Claimed by: GitHub Copilot
Type: task
Blocked by: 03, 04

Create the selected-model export notebook and generate the final package from
the baseline NPZ, selected trials, and selected checkpoints.

Acceptance:

- Exactly Triplet, ArcFace, and SupCon selected checkpoints are exported.
- Each selected NPZ passes the common artifact contract.
- Stable filenames are used with the required NPZ keys.
- Handoff manifest records schema version, checksums, environment, selected IDs,
  data counts, and embedding invariants.
- Selected checkpoints are not copied into the package.
- Export is independently rerunnable from persistent files.

References:

- [spec](../spec.md)
- [handoff design](../../../docs/face-name-metric-learning-handoff.md)

Evidence:

- Added `03_export_selected_embeddings.ipynb` with cleared outputs and fail-fast,
  persistent-file inputs for exactly the three selected trials.
- Added public selected-export and persisted-environment manifest helpers.
- Verified stable NPZ contracts with synthetic HDF5/checkpoint and model-loader
  fixtures without training or requiring GPU checkpoints.
- `python -m unittest discover -s tests -v` passed all 34 tests.

## Comments

- Selected checkpoints remain below ignored `models/metric_learning/` and only
  their validation/test embeddings enter the handoff package.
- Synchronized with the latest local integration tip
  `41fa98915b4b9d266aed26e8bcf4ee7732d172c3`.
- Merged the synchronized implementation with `--no-ff` as `6f8be5b`.
- Re-ran the complete integration unit suite after merge:
  `python -m unittest discover -s tests -v` (34 tests passed).

