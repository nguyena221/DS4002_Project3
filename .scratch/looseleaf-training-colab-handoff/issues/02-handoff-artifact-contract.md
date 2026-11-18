# Handoff artifact contract and integrity I/O

Status: claimed
Type: task
Blocked by:

Implement public utilities for validating, writing, and loading the four-model
handoff package. Keep these utilities independent of model-training internals
so they are usable from notebooks and tests.

Acceptance:

- Validate manifest/HDF5 identity and split coverage.
- Write/read required NPZ keys, shapes, dtypes, finite values, unit norms, and unique indices.
- Align model arrays by `h5_index`, not row position.
- Validate a trial ledger containing exactly 10 candidates and one selected model per loss.
- Build and validate schema-version-1 handoff metadata, checksums, selected IDs,
  split counts, and metric definitions.
- Use fail-fast errors with field-specific messages.

References:

- [spec](../spec.md)
- [handoff design](../../../docs/face-name-metric-learning-handoff.md)

