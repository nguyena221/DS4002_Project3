"""Reproducible stratified image-level bootstrap utilities."""

import numpy as np

from .metrics import pairwise_cosine_similarity, weighted_triplet_accuracy


def stratified_bootstrap_multiplicities(first_names, *, replicates=2000, seed=4002):
    """Sample rows within each first-name stratum and return draw counts."""
    first_names = np.asarray(first_names)
    if first_names.ndim != 1 or first_names.size == 0:
        raise ValueError("first_names must be a nonempty one-dimensional array")
    if not isinstance(replicates, int) or isinstance(replicates, bool) or replicates <= 0:
        raise ValueError("replicates must be a positive integer")

    weights = np.zeros((replicates, first_names.size), dtype=np.int64)
    rng = np.random.default_rng(seed)
    for name in np.unique(first_names):
        indices = np.flatnonzero(first_names == name)
        draws = rng.choice(indices, size=(replicates, indices.size), replace=True)
        for replicate in range(replicates):
            weights[replicate, indices] = np.bincount(
                draws[replicate], minlength=first_names.size
            )[indices]
    return weights


def paired_bootstrap_statistics(
    baseline_embeddings,
    comparison_embeddings,
    h5_index,
    first_names,
    *,
    replicates=2000,
    seed=4002,
    confidence_level=0.95,
):
    """Compute paired model accuracies and a similarity-lift interval."""
    if (
        not isinstance(confidence_level, (int, float))
        or isinstance(confidence_level, bool)
        or not 0 < confidence_level < 1
    ):
        raise ValueError("confidence_level must be strictly between zero and one")
    baseline_embeddings = np.asarray(baseline_embeddings)
    comparison_embeddings = np.asarray(comparison_embeddings)
    h5_index = np.asarray(h5_index)
    if baseline_embeddings.shape != comparison_embeddings.shape:
        raise ValueError("baseline and comparison embeddings must have equal shapes")
    if h5_index.ndim != 1 or h5_index.shape[0] != baseline_embeddings.shape[0]:
        raise ValueError("h5_index must contain one value per embedding row")

    baseline_similarities = pairwise_cosine_similarity(baseline_embeddings)
    comparison_similarities = pairwise_cosine_similarity(comparison_embeddings)
    draw_weights = stratified_bootstrap_multiplicities(
        first_names, replicates=replicates, seed=seed
    )
    baseline_accuracies = np.empty(replicates, dtype=np.float64)
    comparison_accuracies = np.empty(replicates, dtype=np.float64)
    for replicate, multiplicities in enumerate(draw_weights):
        baseline_accuracies[replicate] = weighted_triplet_accuracy(
            baseline_similarities, h5_index, first_names, multiplicities
        )
        comparison_accuracies[replicate] = weighted_triplet_accuracy(
            comparison_similarities, h5_index, first_names, multiplicities
        )

    lifts = comparison_accuracies - baseline_accuracies
    tail = (1 - confidence_level) / 2
    return {
        "baseline_triplet_accuracies": baseline_accuracies,
        "comparison_triplet_accuracies": comparison_accuracies,
        "similarity_lifts": lifts,
        "confidence_interval": np.quantile(lifts, [tail, 1 - tail]),
    }
