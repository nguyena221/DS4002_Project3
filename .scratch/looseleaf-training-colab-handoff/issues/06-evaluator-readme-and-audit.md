# Evaluator README and package audit

Status: resolved
Type: task
Blocked by: 05

Produce the concise teammate-facing handoff README and a local-only audit
workflow that validates the complete package without requiring model weights.

Acceptance:

- README includes upload list, loading example, identity join, metric definitions,
  selected IDs, integrity assertions, bootstrap specification, and expected outputs.
- Audit reloads all four NPZ artifacts and validates checksums, indices, schemas,
  norms, selections, and at least one deterministic metric check.
- Audit writes pass/fail status and timestamp into the handoff manifest.
- Audit notebook remains excluded from version control.

References:

- [spec](../spec.md)
- [handoff design](../../../docs/face-name-metric-learning-handoff.md)

## Answer

Added the evaluator handoff README, a public weights-free package audit helper,
synthetic contract tests, and the ignored local audit notebook. The audit
reloads and aligns all artifacts, validates the manifest contract and checksums,
replays a deterministic metric check, and persists pass/fail audit metadata.

