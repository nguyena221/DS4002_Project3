# Candidate fitting and model selection

Status: claimed
Type: task
Blocked by: 01, 02

Implement the candidate-training notebook and the reusable training behavior for
Triplet Margin, ArcFace, and SupCon using the agreed 3/4/3 candidate grids.

Acceptance:

- Exactly 10 candidate fits with one root-seed-derived child stream per fitting concern.
- 16-by-8 name-balanced batches with 8-by-8 fallback and recorded observed counts.
- Triplet unrestricted random negatives, ArcFace 100-class head, and SupCon same-name positives.
- Training-only flip, Adam 1e-5, AMP, 15-epoch maximum, and four-epoch early stop.
- Exhaustive blocked validation triplet accuracy selects the best epoch.
- Deterministic candidate IDs and selection tie breaks.
- Best checkpoints remain under ignored local-only model storage.
- Trial ledger records every required field and one selected fit per loss.

References:

- [spec](../spec.md)
- [handoff design](../../../docs/face-name-metric-learning-handoff.md)

