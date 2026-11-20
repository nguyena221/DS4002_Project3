# `face_name_metrics`

Python utilities for triplet accuracy calculation and statistical analysis tools for embedding comparison during validation and test.

## Metric Definition

Triplet accuracy (as $\frac{\text{Correct Triplets}}{\text{Total Triplets}}$) over all valid $(a, p, n)$ triplets where $p$ shares $a$'s `first_name` (but has a distinct `h5_index` so no $a$ has the same image as a $p$) and $n$ has a different `first_name`. A triplet is correct if and only if $\cos(a, p) > \cos(a, n)$ (ties are considered incorrect). Simularity lift is then calculated as $(\text{Candidate Triplet Accuracy} - \text{Baseline Triplet Accuracy})$, and a 95% confidence interval over this metric is used as the final datapoint for evaluation.

## Modules

| File | Purpose |
| :--- | :--- |
| `__init__.py` | Public API exports. |
| `metrics.py` | Input validation, L2 normalization, blocked/direct/weighted triplet accuracy, and pairwise cosine similarity. |
| `bootstrap.py` | Stratified image-level resampling and paired similarity-lift confidence intervals. |

## API

* `blocked_triplet_accuracy(embeddings, h5_index, first_names, *, block_size=1024)`: Exhaustive triplet accuracy using anchor blocks and binary search over sorted negatives.
* `direct_triplet_accuracy(embeddings, h5_index, first_names)`: Exact brute-force enumeration oracle for small-sample verification.
* `pairwise_cosine_similarity(embeddings)`: Full $N \times N$ cosine similarity matrix for bootstrap reuse.
* `weighted_triplet_accuracy(similarities, h5_index, first_names, multiplicities)`: Triplet accuracy evaluated under integer row-multiplicity weights.
* `stratified_bootstrap_multiplicities(first_names, *, replicates=2000, seed=4002)`: In-class resampling with replacement returning `(replicates, N)` draw counts.
* `paired_bootstrap_statistics(baseline_embeddings, comparison_embeddings, h5_index, first_names, ...)`: Paired replicate accuracies, similarity lifts (`comparison - baseline`), and percentile confidence intervals (default 95%).