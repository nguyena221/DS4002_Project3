# Face-Name Metric Learning

This project fine-tunes facial embeddings to study whether faces associated with
the same first name become more similar than faces associated with different
names.

## Language

**Candidate configuration**:
A complete, deterministic training specification consisting of a loss, its
hyperparameters, optimizer settings, seed, and training controls.
_Avoid_: Hyperparameter, model (when referring to the full configuration)

**Candidate fit**:
One stochastic training trajectory produced from a single candidate
configuration and seed.
_Avoid_: Repeated run, experiment

**Trial**:
The recorded result of one candidate fit, including its trial ID, best epoch,
validation metric, and selection status.
_Avoid_: Model

**Selected model**:
The best-epoch candidate fit chosen for a loss using exhaustive validation
triplet accuracy and the documented tie-break rules.
_Avoid_: Best model

**Triplet accuracy**:
The fraction of valid ordered `(anchor, positive, negative)` comparisons for
which `cos(anchor, positive) > cos(anchor, negative)`.
_Avoid_: Similarity lift (for the absolute score)

**Similarity lift**:
The difference between a fine-tuned model's triplet accuracy and the pretrained
baseline's triplet accuracy, reported in percentage points.
_Avoid_: Relative improvement, lift ratio

**Handoff artifact**:
A generated file in `artifacts/face_name_metric_learning/` that the evaluation
stage consumes or audits: an embedding file, trial table, manifest, or handoff
README.
_Avoid_: Training artifact

**Positive**:
A face-image row with the same `first_name` as the anchor but a different
original `h5_index`.
_Avoid_: Match (unless "same name" is explicitly intended)

**Negative**:
A face-image row with a different `first_name` from the anchor.
_Avoid_: Impostor, nonmatch

**Bootstrap replicate**:
One seeded, stratified image-level resample used to recompute paired baseline
and fine-tuned evaluation statistics.
_Avoid_: Training repeat
