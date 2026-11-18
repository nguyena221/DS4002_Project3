# Export selected models and handoff manifest

Status: ready-for-agent
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

