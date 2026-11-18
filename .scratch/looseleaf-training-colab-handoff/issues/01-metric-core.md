# Metric core and blocked exhaustive evaluation

Status: resolved
Type: task
Blocked by:

Implement the public behavior needed to compute exhaustive triplet accuracy
from normalized embeddings and `h5_index`/name metadata, plus paired weighted
bootstrap statistics. Use small fixtures for correctness and a blocked
implementation for scalability.

Acceptance:

- Strict `cos(anchor, positive) > cos(anchor, negative)` semantics; ties are incorrect.
- Positives share first name and have distinct original `h5_index`.
- Negatives have a different first name.
- Blocked similarity-matrix evaluation agrees with direct enumeration on small fixtures.
- Matrix-based multiplicity weights agree with direct bootstrap enumeration.
- Repeated bootstrap seeds are reproducible and duplicate original IDs cannot self-pair.

References:

- [spec](../spec.md)
- [handoff design](../../../docs/face-name-metric-learning-handoff.md)
- [glossary](../../../GLOSSARY.md)

Comments:

- Integrated as merge commit `93a5740`; implementation commit `a82a303`.
  Verified with `python -m unittest discover -s tests -v` (6 tests passed).

