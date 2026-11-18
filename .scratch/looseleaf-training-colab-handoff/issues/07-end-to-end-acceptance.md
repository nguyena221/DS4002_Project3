# Fresh-environment handoff acceptance

Status: ready-for-agent
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
