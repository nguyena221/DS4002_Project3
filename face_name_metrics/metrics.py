"""Blocked exhaustive triplet metric implementation."""

import numpy as np


def _validated_inputs(embeddings, h5_index, first_names):
    embeddings = np.asarray(embeddings)
    h5_index = np.asarray(h5_index)
    first_names = np.asarray(first_names)
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a two-dimensional array")
    row_count = embeddings.shape[0]
    if h5_index.ndim != 1 or h5_index.shape[0] != row_count:
        raise ValueError("h5_index must contain one value per embedding row")
    if first_names.ndim != 1 or first_names.shape[0] != row_count:
        raise ValueError("first_names must contain one value per embedding row")
    if np.unique(h5_index).size != row_count:
        raise ValueError("h5_index values must be unique")
    if not np.issubdtype(embeddings.dtype, np.number):
        raise ValueError("embeddings must be numeric")
    if not np.isfinite(embeddings).all():
        raise ValueError("embeddings must contain only finite values")
    norms = np.linalg.norm(embeddings, axis=1)
    if np.any(norms == 0):
        raise ValueError("embedding rows must have nonzero L2 norm")
    return embeddings / norms[:, None], h5_index, first_names


def blocked_triplet_accuracy(
    embeddings, h5_index, first_names, *, block_size=1024
):
    """Return exhaustive ordered-triplet accuracy using strict cosine comparison.

    Similarity rows are materialized one anchor block at a time. Sorting the
    negative similarities lets each positive's correct comparison count be
    obtained without constructing Python triplet objects.
    """
    if not isinstance(block_size, int) or isinstance(block_size, bool) or block_size <= 0:
        raise ValueError("block_size must be a positive integer")
    normalized, h5_index, first_names = _validated_inputs(
        embeddings, h5_index, first_names
    )
    correct = 0
    total = 0
    row_count = normalized.shape[0]

    for start in range(0, row_count, block_size):
        stop = min(start + block_size, row_count)
        similarities = normalized[start:stop] @ normalized.T
        for anchor_offset, anchor in enumerate(range(start, stop)):
            positive = (first_names == first_names[anchor]).copy()
            positive[anchor] = False
            negative = first_names != first_names[anchor]
            positive_similarities = similarities[anchor_offset, positive]
            negative_similarities = np.sort(similarities[anchor_offset, negative])
            if positive_similarities.size == 0 or negative_similarities.size == 0:
                continue
            correct += int(
                np.searchsorted(
                    negative_similarities, positive_similarities, side="left"
                ).sum()
            )
            total += int(
                positive_similarities.size * negative_similarities.size
            )

    if total == 0:
        raise ValueError("metadata must define at least one valid ordered triplet")
    return correct / total


def direct_triplet_accuracy(embeddings, h5_index, first_names):
    """Enumerate a small fixture directly as an independent reference oracle."""
    normalized, h5_index, first_names = _validated_inputs(
        embeddings, h5_index, first_names
    )
    correct = 0
    total = 0
    for anchor in range(normalized.shape[0]):
        positive_indices = np.flatnonzero(
            (first_names == first_names[anchor]) & (h5_index != h5_index[anchor])
        )
        negative_indices = np.flatnonzero(first_names != first_names[anchor])
        for positive in positive_indices:
            positive_similarity = np.dot(
                normalized[anchor], normalized[positive]
            )
            for negative in negative_indices:
                total += 1
                if positive_similarity > np.dot(
                    normalized[anchor], normalized[negative]
                ):
                    correct += 1
    if total == 0:
        raise ValueError("metadata must define at least one valid ordered triplet")
    return correct / total


def pairwise_cosine_similarity(embeddings):
    """Return the full pairwise cosine matrix for bootstrap reuse."""
    embeddings = np.asarray(embeddings)
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a two-dimensional array")
    if not np.issubdtype(embeddings.dtype, np.number):
        raise ValueError("embeddings must be numeric")
    if not np.isfinite(embeddings).all():
        raise ValueError("embeddings must contain only finite values")
    norms = np.linalg.norm(embeddings, axis=1)
    if np.any(norms == 0):
        raise ValueError("embedding rows must have nonzero L2 norm")
    normalized = embeddings / norms[:, None]
    return normalized @ normalized.T


def weighted_triplet_accuracy(
    similarities, h5_index, first_names, multiplicities
):
    """Evaluate a bootstrap resample using original-row multiplicity weights."""
    similarities = np.asarray(similarities)
    h5_index = np.asarray(h5_index)
    first_names = np.asarray(first_names)
    multiplicities = np.asarray(multiplicities)
    row_count = h5_index.size
    if h5_index.ndim != 1:
        raise ValueError("h5_index must be a one-dimensional array")
    if np.unique(h5_index).size != row_count:
        raise ValueError("h5_index values must be unique")
    if similarities.shape != (row_count, row_count):
        raise ValueError("similarities must be square and match h5_index")
    if not np.isfinite(similarities).all():
        raise ValueError("similarities must contain only finite values")
    if first_names.ndim != 1 or first_names.shape[0] != row_count:
        raise ValueError("first_names must contain one value per similarity row")
    if multiplicities.ndim != 1 or multiplicities.shape[0] != row_count:
        raise ValueError("multiplicities must contain one weight per similarity row")
    if not np.issubdtype(multiplicities.dtype, np.integer):
        raise ValueError("multiplicities must be integers")
    if np.any(multiplicities < 0):
        raise ValueError("multiplicities cannot be negative")

    correct = 0
    total = 0
    for anchor in np.flatnonzero(multiplicities):
        positive = (first_names == first_names[anchor]).copy()
        positive[anchor] = False
        positive &= multiplicities > 0
        negative = (first_names != first_names[anchor]) & (multiplicities > 0)
        if not positive.any() or not negative.any():
            continue

        negative_order = np.argsort(similarities[anchor, negative], kind="stable")
        sorted_negative = similarities[anchor, negative][negative_order]
        sorted_negative_weights = multiplicities[negative][negative_order]
        cumulative_negative_weights = np.cumsum(sorted_negative_weights)

        positive_similarities = similarities[anchor, positive]
        lower_counts = np.searchsorted(
            sorted_negative, positive_similarities, side="left"
        )
        weighted_lower = np.where(
            lower_counts > 0,
            np.take(
                cumulative_negative_weights,
                np.maximum(lower_counts - 1, 0),
            ),
            0,
        )
        anchor_weight = int(multiplicities[anchor])
        correct += anchor_weight * int(
            np.dot(multiplicities[positive], weighted_lower)
        )
        total += anchor_weight * int(multiplicities[positive].sum()) * int(
            multiplicities[negative].sum()
        )

    if total == 0:
        raise ValueError("multiplicities must define at least one valid triplet")
    return correct / total
