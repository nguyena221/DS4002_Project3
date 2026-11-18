import unittest

import numpy as np

from face_name_metrics import (
    blocked_triplet_accuracy,
    direct_triplet_accuracy,
    paired_bootstrap_statistics,
    pairwise_cosine_similarity,
    stratified_bootstrap_multiplicities,
    weighted_triplet_accuracy,
)


class TripletAccuracyTests(unittest.TestCase):
    def test_blocked_and_direct_accuracy_agree_for_small_fixture(self):
        embeddings = np.asarray(
            [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [0.8, -0.6]]
        )
        h5_index = np.asarray([101, 102, 103, 201])
        first_names = np.asarray(["Ada", "Ada", "Grace", "Grace"])

        direct = direct_triplet_accuracy(embeddings, h5_index, first_names)

        self.assertAlmostEqual(
            blocked_triplet_accuracy(
                embeddings, h5_index, first_names, block_size=1
            ),
            direct,
        )

    def test_blocked_accuracy_is_strict_and_matches_direct_geometry(self):
        embeddings = np.asarray(
            [
                [1.0, 0.0],
                [0.8, 0.6],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.6, 0.8],
                [0.0, 1.0],
            ]
        )
        h5_index = np.asarray([11, 12, 13, 21, 22, 23])
        first_names = np.asarray(["Ada", "Ada", "Grace", "Linus", "Linus", "Ada"])

        accuracy = blocked_triplet_accuracy(
            embeddings, h5_index, first_names, block_size=2
        )

        self.assertAlmostEqual(accuracy, 3 / 13)

    def test_equal_positive_and_negative_cosines_are_incorrect(self):
        embeddings = np.asarray(
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
        )
        h5_index = np.asarray([1, 2, 3])
        first_names = np.asarray(["Ada", "Ada", "Grace"])

        self.assertEqual(
            direct_triplet_accuracy(embeddings, h5_index, first_names), 0.0
        )


class BootstrapMetricTests(unittest.TestCase):
    def test_paired_statistics_share_draws_and_report_lift_interval(self):
        ada_angles = np.linspace(-0.15, 0.15, 8)
        grace_angles = np.pi / 2 + np.linspace(-0.15, 0.15, 8)
        embeddings = np.vstack(
            [
                np.column_stack([np.cos(ada_angles), np.sin(ada_angles)]),
                np.column_stack([np.cos(grace_angles), np.sin(grace_angles)]),
            ]
        )
        tighter_ada = ada_angles * 0.3
        tighter_grace = np.pi / 2 + (grace_angles - np.pi / 2) * 0.3
        comparison = np.vstack(
            [
                np.column_stack([np.cos(tighter_ada), np.sin(tighter_ada)]),
                np.column_stack(
                    [np.cos(tighter_grace), np.sin(tighter_grace)]
                ),
            ]
        )
        h5_index = np.arange(10, 26)
        first_names = np.asarray(["Ada"] * 8 + ["Grace"] * 8)

        first = paired_bootstrap_statistics(
            embeddings,
            comparison,
            h5_index,
            first_names,
            replicates=20,
            seed=19,
        )
        second = paired_bootstrap_statistics(
            embeddings,
            comparison,
            h5_index,
            first_names,
            replicates=20,
            seed=19,
        )

        np.testing.assert_array_equal(first["similarity_lifts"], second["similarity_lifts"])
        np.testing.assert_allclose(
            first["similarity_lifts"],
            first["comparison_triplet_accuracies"]
            - first["baseline_triplet_accuracies"],
        )
        expected = np.quantile(first["similarity_lifts"], [0.025, 0.975])
        np.testing.assert_allclose(first["confidence_interval"], expected)

    def test_seeded_bootstrap_preserves_name_strata_reproducibly(self):
        first_names = np.asarray(["Ada"] * 4 + ["Grace"] * 2)

        first = stratified_bootstrap_multiplicities(
            first_names, replicates=7, seed=4002
        )
        second = stratified_bootstrap_multiplicities(
            first_names, replicates=7, seed=4002
        )

        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first[:, :4].sum(axis=1), np.full(7, 4))
        np.testing.assert_array_equal(first[:, 4:].sum(axis=1), np.full(7, 2))

    def test_weighted_accuracy_matches_explicit_resample_without_self_pairs(self):
        embeddings = np.asarray(
            [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [0.6, 0.8]]
        )
        h5_index = np.asarray([10, 11, 20, 21])
        first_names = np.asarray(["Ada", "Ada", "Grace", "Grace"])
        multiplicities = np.asarray([2, 1, 0, 3])
        similarities = pairwise_cosine_similarity(embeddings)

        sampled_indices = np.repeat(
            np.arange(len(multiplicities)), multiplicities
        )
        correct = 0
        total = 0
        for anchor in sampled_indices:
            for positive in sampled_indices:
                if (
                    first_names[positive] != first_names[anchor]
                    or h5_index[positive] == h5_index[anchor]
                ):
                    continue
                for negative in sampled_indices:
                    if first_names[negative] == first_names[anchor]:
                        continue
                    total += 1
                    correct += int(
                        similarities[anchor, positive]
                        > similarities[anchor, negative]
                    )

        self.assertAlmostEqual(
            weighted_triplet_accuracy(
                similarities, h5_index, first_names, multiplicities
            ),
            correct / total,
        )


if __name__ == "__main__":
    unittest.main()
