# LooseLeaf Training and Colab Evaluation Handoff

Status: implementation-complete-production-not-executed

Implementation status: the training, export, validation, and audit code is
implemented and covered by synthetic contract tests. Production GPU candidate
fitting, production embedding export, and the independent audit have not been
executed. The repository therefore does not yet contain a production
`handoff_manifest.json`, `trial_results.csv`, or the four production NPZ files.

## Problem Statement

The project must perform GPU-intensive face embedding fine-tuning on UVA
LooseLeaf and then hand the resulting outputs to a teammate for independent
evaluation in Google Colab. The project currently has preprocessing outputs and
a written handoff design, but not the executable notebooks, reproducible
training records, selected-model embeddings, or evaluator-facing contract
needed to complete that transfer reliably.

Without a compact, self-describing handoff, the evaluator could use the wrong
trial, mismatch embeddings with image identities, recompute metrics from
different splits, depend on unavailable checkpoints, or be unable to reproduce
the model selection and bootstrap analysis.

## Solution

Create a LooseLeaf training pipeline that begins with the prepared face dataset,
trains and selects candidate fits for Triplet Margin, ArcFace, and Supervised
Contrastive learning, and produces four selected embedding artifacts: the
pretrained IResNet-100 baseline and one fine-tuned selected model per loss.

Package those embeddings with a trial ledger, integrity manifest, and concise
Colab instructions. Keep large checkpoints local and unnecessary, keep the
handoff independently rerunnable, and allow the evaluator to compute exhaustive
triplet accuracy, similarity lift, and paired stratified bootstrap confidence
intervals without access to LooseLeaf, original images, or model checkpoints.

The primary acceptance seam is the generated handoff package as consumed by a
fresh Colab environment. It should be possible to verify the feature entirely
from package inputs and observable outputs rather than by inspecting training
implementation details.

## User Stories

1. As a training operator, I want to verify the LooseLeaf environment before training, so that GPU, package, and pretrained-model problems fail before expensive work begins.
2. As a training operator, I want the setup stage to report Python, PyTorch, CUDA, GPU, pytorch-metric-learning, and h5py versions, so that the run can be reproduced.
3. As a training operator, I want the setup stage to assert that the processed dataset and manifest exist, so that missing local data is reported immediately.
4. As a training operator, I want the setup stage to verify the actual split and class counts, so that stale estimates cannot silently change the candidate-fitting plan.
5. As a training operator, I want the setup stage to verify that every manifest identity maps to one valid original dataset row, so that identity joins cannot drift.
6. As a training operator, I want a one-image pretrained-model smoke test, so that I know the model loads and produces a 512-dimensional embedding before starting candidate fits.
7. As a training operator, I want a single visible configuration section, so that changes to seeds, candidate grids, and training controls are easy to review.
8. As a training operator, I want the root seed to produce named child RNG streams, so that batch selection, augmentation, loss sampling, and evaluation resampling are independently reproducible.
9. As a training operator, I want three Triplet Margin candidate configurations at margins 0.1, 0.2, and 0.4, so that the loss's margin is tuned.
10. As a training operator, I want four ArcFace candidate configurations over margins 0.2 and 0.4 and scales 30 and 64, so that both angular margin and scale are tuned.
11. As a training operator, I want three SupCon candidate configurations at temperatures 0.07, 0.10, and 0.20, so that the contrastive temperature is tuned.
12. As a training operator, I want exactly one stochastic candidate fit per candidate configuration, so that the planned search trains 10 candidate fits rather than undocumented additional fits.
13. As a training operator, I want each candidate configuration to have a deterministic trial ID, so that its checkpoint, trial row, and selected status can be traced unambiguously.
14. As a training operator, I want every candidate fit to fine-tune all IResNet-100 backbone parameters, so that each loss may subtly adjust pretrained facial representations.
15. As a training operator, I want Adam configured with learning rate 1e-5, betas 0.9 and 0.999, and weight decay 0, so that optimization follows the agreed protocol.
16. As a training operator, I want CUDA automatic mixed precision during training, so that LooseLeaf GPU memory and throughput are used efficiently.
17. As a training operator, I want exported embeddings to be converted to float32, so that Colab results do not depend on mixed-precision types.
18. As a training operator, I want name-balanced batches of 16 names by 8 images, so that each anchor has same-name positives and a broad set of different-name negatives.
19. As a training operator, I want a documented 8-by-8 fallback batch composition, so that limited GPU memory produces an explicit controlled adjustment.
20. As a training operator, I want the sampler to choose different source image rows for the repeated images in each batch, so that a batch cannot contain duplicate-original positives by accident.
21. As a training operator, I want an actual count of steps and image draws per epoch, so that claims about epoch coverage are truthful.
22. As a training operator, I want Triplet Margin to sample positives uniformly from same-name batch members and negatives uniformly from different-name batch members, so that the run does not accidentally perform hard-negative mining.
23. As a training operator, I want ArcFace to create and optimize its temporary 100-class classifier head, so that angular-margin training is mathematically valid.
24. As a training operator, I want the temporary ArcFace classifier head excluded from the evaluation package, so that the teammate only receives what is needed for embedding evaluation.
25. As a training operator, I want the SupCon implementation and sampler behavior recorded, so that same-name positive construction can be reproduced.
26. As a training operator, I want only a 50% random horizontal flip during training, so that fine-tuning uses the agreed mild augmentation.
27. As a training operator, I want validation and test images evaluated without augmentation, so that exported representations are deterministic for each source row.
28. As a training operator, I want each candidate fit to run for no more than 15 epochs, so that the compute budget remains bounded.
29. As a training operator, I want a candidate fit to stop after four epochs without at least 1e-4 validation improvement, so that non-improving fits do not consume unnecessary GPU time.
30. As a training operator, I want validation embeddings recomputed once per validation epoch, so that exhaustive selection does not repeatedly forward-pass the same dataset unnecessarily.
31. As a training operator, I want exhaustive validation triplet accuracy computed with blocked matrix operations, so that the 8.9 billion valid comparisons do not have to be materialized as Python objects.
32. As a training operator, I want strict cosine comparison semantics, so that ties are counted as incorrect.
33. As a training operator, I want the best epoch selected by maximum triplet accuracy, so that model selection optimizes the final comparison metric rather than training loss.
34. As a training operator, I want deterministic hyperparameter tie-breaking, so that equal validation scores cannot produce selection dependent on file or directory order.
35. As a training operator, I want each completed candidate fit to produce one trial row, so that the search is auditable without retaining every optimizer log.
36. As a training operator, I want each trial row to record configuration, seed, epochs, selection metric, early stopping, workload counts, device, selected status, and notes, so that a reviewer can understand and reproduce the fit.
37. As a training operator, I want only the best-epoch checkpoint retained for each candidate fit, so that local storage is bounded.
38. As a training operator, I want checkpoints kept outside Git, so that the repository remains reviewable and cloneable.
39. As a project maintainer, I want exactly one selected model per fine-tuning loss, so that the evaluator has a clear final model for each objective.
40. As a project maintainer, I want the selected candidate fit used directly for test evaluation, so that validation is not retroactively merged into training and the split boundary remains intact.
41. As an embedding exporter, I want every selected model export to use original h5 indices, so that embeddings can be aligned despite any row-order difference.
42. As an embedding exporter, I want each selected embedding package to contain separate validation and test arrays, so that split leakage is visible and preventable.
43. As an evaluator, I want baseline and selected-model embeddings keyed by h5_index rather than order, so that I can safely join and compare independently exported arrays.
44. As an evaluator, I want every embedding array to be finite, float32, 512-dimensional, and L2-normalized, so that matrix products compute cosine similarity consistently.
45. As an evaluator, I want the embedding package to cover every validation and test h5 index exactly once, so that missing or duplicate rows cannot bias evaluation.
46. As an evaluator, I want to load all evaluation inputs without original images, so that Colab requires only lightweight identifiers and embeddings.
47. As an evaluator, I want the handoff to identify exactly four evaluation models, so that I know to use baseline, Triplet Margin, ArcFace, and SupCon without guessing among trials.
48. As an evaluator, I want the selected trial IDs exposed in the handoff manifest, so that I can distinguish selected results from the complete trial history.
49. As an evaluator, I want checksums for dataset, manifest, trial ledger, and embedding artifacts, so that I can detect a partially copied or corrupted handoff.
50. As an evaluator, I want package, pretrained-weight, code, and data fingerprints, so that I can identify the exact run behind the exported embeddings.
51. As an evaluator, I want a versioned handoff schema, so that future schema changes are explicit and readable.
52. As an evaluator, I want a short loading example and integrity assertions in the handoff instructions, so that I can begin evaluation without consulting the training author.
53. As an evaluator, I want triplet accuracy defined as the correct fraction of all valid ordered anchor-positive-negative comparisons, so that validation and report metrics use the same meaning.
54. As an evaluator, I want similarity lift defined as fine-tuned triplet accuracy minus pretrained baseline triplet accuracy, so that the project does not confuse an absolute score with an improvement.
55. As an evaluator, I want to report lift in percentage points, so that differences are interpretable and consistently formatted.
56. As an evaluator, I want final Colab evaluation to recompute selected-model validation and test metrics, so that result generation is independent of LooseLeaf selection logs.
57. As an evaluator, I want pairwise similarities computed once per model and split, so that exhaustive comparison does not repeatedly multiply embeddings.
58. As an evaluator, I want triplet totals derived from actual name counts and class masks, so that stale back-of-the-napkin estimates cannot affect the denominator.
59. As an evaluator, I want 2,000 paired image-level bootstrap replicates, so that baseline and fine-tuned models are compared on identical resamples.
60. As an evaluator, I want bootstrap resampling stratified by split and first name, so that random class-frequency variation does not dominate the uncertainty estimate.
61. As an evaluator, I want root and child bootstrap seeds specified, so that all replicates can be regenerated exactly.
62. As an evaluator, I want duplicated bootstrap draws to retain multiplicity as weights, so that the bootstrap remains mathematically faithful without exporting index arrays for every replicate.
63. As an evaluator, I want comparisons between two draws of the same original h5 index excluded, so that a duplicate resampled image cannot compare its embedding with itself as a positive.
64. As an evaluator, I want bootstrap metrics calculated from precomputed similarity matrices with weighted counts, so that 2,000 replicates remain feasible in Colab.
65. As an evaluator, I want a 95% confidence interval for each fine-tuned model's test similarity lift, so that I can identify statistically significant improvement.
66. As an evaluator, I want significance defined as an interval excluding zero, so that the final conclusion follows the agreed rule.
67. As an evaluator, I want a final result table containing model, split, triplet accuracy, similarity lift, and confidence interval, so that the report can compare every selected model consistently.
68. As a project maintainer, I want the local audit notebook to remain uncommitted, so that the repository retains the audit result without retaining disposable QA machinery.
69. As a project maintainer, I want the audit result embedded in the handoff manifest, so that the accepted package records that its checks passed.
70. As a project maintainer, I want the committed notebooks to have cleared outputs, so that large or machine-specific results do not pollute version control.
71. As a project maintainer, I want each notebook to rerun from persistent files rather than retained notebook variables, so that returning to LooseLeaf later does not change whether the pipeline works.
72. As a reviewer, I want failures in data, metric, schema, or selection assertions to stop the pipeline, so that invalid handoff artifacts are not silently generated.

## Implementation Decisions

- The implementation will be split into four committed notebook stages: setup and
  smoke validation, baseline embedding export, candidate fitting and selection,
  and selected-model embedding export. A fifth audit notebook will be local-only.
- Each committed notebook will have a single configuration area, explicit
  persistent inputs and outputs, cleared outputs, and fail-fast assertions.
- The testing and handoff seam will be the completed artifact package. A fresh
  evaluator loads that package exactly as Colab will, joins embeddings by stable
  image identity, and observes the exported data and summary records.
- The prepared face dataset and manifest remain the source of truth. Runtime
  code derives counts and validates them against the documented expected
  values.
- The training search consists of 10 candidate configurations: three Triplet
  Margin, four ArcFace, and three SupCon. Each configuration has one
  root-seeded candidate fit.
- The root seed is 4002. Named child seeds are derived for sampler, augmentation,
  loss-level sampling, and bootstrap behavior; exhaustive evaluation is
  deterministic.
- Candidate fitting uses all IResNet-100 backbone parameters, Adam at 1e-5,
  CUDA automatic mixed precision, name-balanced 16-by-8 batches where possible,
  8-by-8 fallback, and training-only horizontal flipping at probability 0.5.
- Triplet Margin uses unrestricted in-batch random negative sampling.
  ArcFace adds a temporary 100-class classifier head. SupCon uses same-name
  examples as positives according to the pinned library implementation.
- A candidate fit runs at most 15 epochs and may stop after four epochs without
  validation improvement of at least 1e-4.
- Exhaustive validation triplet accuracy is the only model-selection metric.
  The criterion is strict cosine greater-than, and ties are incorrect.
- Selection considers best-epoch validation triplet accuracy first, then smaller
  margin or temperature, and finally smaller ArcFace scale.
- One selected model per loss is exported. The selected train-only checkpoint is
  used directly and is never refit on train plus validation.
- The trial ledger contains one compact row per candidate fit. Batch logs,
  TensorBoard events, optimizer-state history, and every epoch's similarity
  matrix are transient and out of scope for version control.
- The baseline and selected models each export validation and test embedding
  arrays keyed by 64-bit original image IDs. Embeddings are finite, normalized
  float32 arrays with exactly 512 dimensions.
- Shared metadata comes from the existing manifest rather than being duplicated
  in every embedding package.
- The handoff manifest uses schema version 1 and records selected trial IDs,
  checksums, versions, environment details, data counts, metric definitions,
  embedding invariants, and final audit status.
- Checkpoints are local implementation material and are excluded from Git. The
  teammate does not need them to run evaluation.
- The evaluator consumes four model artifact packages plus shared handoff
  records, aligns rows by original image ID, and computes similarity matrices in
  blocks if necessary.
- Colab performs 2,000 paired, name-stratified image-level bootstrap replicates.
  Multiplicity weights are applied to the precomputed similarity matrix rather
  than reconstructing billions of triplets per replicate.
- A valid positive requires the same first name and distinct original image
  IDs. Duplicate bootstrap draws of one original image are excluded from
  positive pair formation with each other.
- Triplet accuracy is an absolute fraction. Similarity lift is the signed
  percentage-point difference from the pretrained baseline.

## Testing Decisions

- A good test verifies externally observable behavior of a pipeline stage or the
  generated handoff package. Tests do not inspect sampler internals, optimizer
  state shape, private library calls, or notebook implementation.
- The primary test subject is the handoff package as a fresh evaluator sees it:
  its model set, schemas, stable identities, split coverage, dtypes, dimensions,
  normalization, checksums, trial selections, and metric definitions.
- Data-contract tests verify exact training, validation, and test counts; unique
  image identities; complete per-split coverage; 100 first-name classes; and a
  successful mapping between embeddings and shared metadata.
- Embedding-contract tests verify float32 finite vectors with shape 512,
  L2 norm within the documented tolerance, correct int64 identity arrays,
  exact split membership, and no duplicate source IDs.
- Selection-contract tests verify 10 total candidate rows, the required loss and
  grid coverage, one selected trial for each loss, a selected baseline, and
  deterministic tie behavior on a small synthetic score table.
- Metric tests use small synthetic embeddings with known cosines to verify that
  strict greater-than, ties-as-incorrect, positive identity exclusion, and
  negative-name construction match the documented behavior.
- Exhaustive comparison tests compare the blocked matrix implementation with a
  direct enumeration on a tiny fixture. The tiny fixture is the oracle; the full
  9,660-image run is a scalability path, not a unit-test fixture.
- Bootstrap tests compare a weighted, matrix-based implementation against a
  direct resample on a tiny fixture. They verify paired multiplicities, name
  stratification, reproducible seeds, duplicate-source exclusion, and interval
  generation.
- Reproducibility tests rerun loading and validation from cleared persistent
  inputs, without relying on retained notebook variables.
- The existing preprocessing verification assertions are the closest prior art:
  they use hard failures for data shape, numeric range, identity, and split
  invariants. The new pipeline should follow the same fail-fast style.
- End-to-end acceptance begins from a copied artifact package and a fresh
  Colab-compatible Python environment. It ends with validation and test result
  tables plus lift confidence intervals.

## Out of Scope

- Repeating the existing face detection, age filtering, alignment, or
  70/15/15 splitting pipeline.
- Altering the Names100 preprocessing or adult-age eligibility rules.
- Releasing or committing the raw Names100 dataset, processed HDF5 dataset, or
  any source images.
- Hyperparameter searches beyond the agreed 10 candidate configurations.
- Multiple training seeds per candidate configuration.
- Retraining selected models on combined train and validation data.
- Hard-negative mining, semi-hard-negative mining, identity pair labels, or
  person-level grouping.
- Additional loss functions, model architectures, or embedding dimensions.
- Committing checkpoints, optimizer states, per-batch logs, TensorBoard events,
  or every epoch's similarity matrices.
- Evaluating from face images or pretrained-model weights in Colab.
- Building the final Colab analysis notebook, report plots, or statistical
  interpretation as part of the LooseLeaf training implementation.
- Deploying a hosted model, API, or interactive application.
- Creating a cloud data pipeline, CI workflow, or automatic LooseLeaf job
  scheduler.

## Further Notes

- The measured validation and test splits each contain 9,660 images. The
  expected exhaustive comparison count is 8,916,735,196 valid ordered triplets
  per split, not the earlier estimate of 1.18 billion.
- Similarity lift is a signed percentage-point difference, not a ratio and not
  the absolute triplet-accuracy score.
- The handoff manifest and concise evaluator README are part of the product,
  not optional documentation. A handoff without them is incomplete even if the
  embedding arrays are mathematically valid.
- Stable artifact names simplify teammate usage; versioning belongs in the
  schema and content fingerprints rather than timestamped filenames.
