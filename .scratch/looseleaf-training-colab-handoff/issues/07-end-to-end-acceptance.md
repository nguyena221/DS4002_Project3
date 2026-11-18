# Fresh-environment handoff acceptance

Status: pending-production-artifacts
Claimed by: GitHub Copilot
Type: task
Blocked by: 06

Implement and run the highest-seam acceptance checks against the generated
handoff package as a fresh Colab evaluator would consume it.

Acceptance:

- A clean environment loads only the committed/shared handoff inputs.
- Package schema, selections, counts, checksums, and all embedding invariants pass.
- Tiny fixture tests for metrics and bootstrap pass.
- Blocked exhaustive evaluation completes on a feasible fixture and matches direct enumeration.
- Documentation and notebook paths match the package actually generated.
- All public tests pass and the spec requirements have implementation evidence.

References:

- [spec](../spec.md)
- [handoff design](../../../docs/face-name-metric-learning-handoff.md)

## Answer

Added a public fresh-evaluator helper and synthetic end-to-end acceptance
fixture. Synthetic copied packages validate schemas, selections, checksums,
embedding invariants, direct/blocked metrics, paired bootstrap outputs, and
documented notebook/package paths without source images, model weights, or
`faces.h5`. Production acceptance remains pending because production candidate
fitting, production exports, and the independent audit have not been executed.

## Evidence

- Synthetic contract acceptance is covered by the complete public unit suite:
  `python -m unittest discover -s tests -v` (41 tests passed).
- Synchronized with integration tip
  `8c02ab2ed2577109d93e02ed10ba17324c731e64` (already up to date).
- Merged the synchronized implementation with `--no-ff` as `793c79f`.
- Production readiness is reported as `pending` with the missing production
  trial table, manifest, and four NPZ files enumerated.
- No production embeddings, checkpoints, downloads, or candidate fits were
  generated, so no production pass is claimed.
