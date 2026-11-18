import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from face_name_metric_learning.training import (
    DEFAULT_TRAINING_CONFIG,
    BestEpochSelector,
    NameBalancedBatchPlan,
    candidate_configurations,
    child_seed,
    loss_contract,
    select_candidates,
    shuffled_name_balanced_batches,
    trial_checkpoint_path,
    triplet_indices,
    write_trial_ledger,
    supcon_positive_mask,
)


class CandidateGridContractTests(unittest.TestCase):
    def test_grid_has_exactly_three_four_three_configurations_and_stable_ids(self):
        candidates = candidate_configurations()

        self.assertEqual([item.loss for item in candidates], [
            "triplet_margin",
            "triplet_margin",
            "triplet_margin",
            "arcface",
            "arcface",
            "arcface",
            "arcface",
            "supcon",
            "supcon",
            "supcon",
        ])
        self.assertEqual(
            [item.trial_id for item in candidates],
            [
                "triplet_margin0p1_seed4002",
                "triplet_margin0p2_seed4002",
                "triplet_margin0p4_seed4002",
                "arcface_m0p2_s30_seed4002",
                "arcface_m0p2_s64_seed4002",
                "arcface_m0p4_s30_seed4002",
                "arcface_m0p4_s64_seed4002",
                "supcon_t0p07_seed4002",
                "supcon_t0p1_seed4002",
                "supcon_t0p2_seed4002",
            ],
        )
        self.assertEqual(len({item.trial_id for item in candidates}), 10)
        self.assertEqual(DEFAULT_TRAINING_CONFIG.max_epochs, 15)
        self.assertEqual(DEFAULT_TRAINING_CONFIG.early_stop_patience, 4)
        self.assertEqual(DEFAULT_TRAINING_CONFIG.learning_rate, 1e-5)

    def test_named_child_streams_are_reproducible_and_independent(self):
        candidate = candidate_configurations()[0]

        sampler_seed = child_seed(candidate, "batch-sampling")
        loss_seed = child_seed(candidate, "loss-sampling")

        self.assertEqual(sampler_seed, child_seed(candidate, "batch-sampling"))
        self.assertNotEqual(sampler_seed, loss_seed)
        self.assertNotEqual(
            sampler_seed,
            child_seed(candidate_configurations()[1], "batch-sampling"),
        )


class NameBalancedBatchContractTests(unittest.TestCase):
    def test_primary_and_fallback_plans_record_actual_full_batch_work(self):
        h5_index = np.arange(40, dtype=np.int64)
        labels = np.repeat(np.arange(5), 8)
        rng = np.random.default_rng(9)

        primary = NameBalancedBatchPlan(names_per_batch=5, images_per_name=4)
        batches = list(
            shuffled_name_balanced_batches(h5_index, labels, primary, rng)
        )

        self.assertEqual(len(batches), 2)
        self.assertTrue(all(len(batch) == 20 for batch in batches))
        for batch in batches:
            self.assertEqual(len(np.unique(batch)), len(batch))
            values, counts = np.unique(labels[batch], return_counts=True)
            self.assertEqual(set(values), set(range(5)))
            np.testing.assert_array_equal(counts, np.full(5, 4))

        fallback = NameBalancedBatchPlan(names_per_batch=4, images_per_name=4)
        fallback_batches = list(
            shuffled_name_balanced_batches(
                h5_index, labels, fallback, np.random.default_rng(9)
            )
        )
        self.assertEqual(len(fallback_batches), 2)
        self.assertTrue(all(len(batch) == 16 for batch in fallback_batches))

    def test_real_dataset_workload_drops_only_the_incomplete_tail(self):
        plan = NameBalancedBatchPlan.names_16_by_8()
        steps = NameBalancedBatchPlan.steps_for_images(45_078, plan)

        self.assertEqual(steps, 352)
        self.assertEqual(steps * plan.batch_size, 45_056)

    def test_batching_rejects_classes_without_enough_source_images(self):
        with self.assertRaisesRegex(ValueError, "images_per_name"):
            list(
                shuffled_name_balanced_batches(
                    np.arange(4),
                    np.array([0, 0, 1, 1]),
                    NameBalancedBatchPlan(2, 3),
                    np.random.default_rng(1),
                )
            )


class LossSamplingContractTests(unittest.TestCase):
    def test_triplet_sampler_uses_unrestricted_random_other_name_negatives(self):
        batch_h5_index = np.arange(8)
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])

        anchor, positive, negative = triplet_indices(
            batch_h5_index, labels, np.random.default_rng(4002)
        )

        np.testing.assert_array_equal(anchor, np.arange(8))
        self.assertTrue(
            np.all(labels[anchor] == labels[positive])
            and np.all(batch_h5_index[anchor] != batch_h5_index[positive])
        )
        self.assertTrue(np.all(labels[anchor] != labels[negative]))

    def test_loss_contracts_cover_random_triplets_arcface_head_and_supcon(self):
        candidates = candidate_configurations()
        contracts = {candidate.loss: loss_contract(candidate) for candidate in candidates}

        self.assertIn("unrestricted", contracts["triplet_margin"]["negative_sampling"])
        self.assertEqual(contracts["arcface"]["classifier_classes"], 100)
        self.assertTrue(contracts["arcface"]["temporary_head"])
        self.assertIn("same-name", contracts["supcon"]["positives"])

    def test_supcon_positives_are_same_name_and_exclude_self(self):
        mask = supcon_positive_mask(np.array([0, 0, 1, 1]))
        np.testing.assert_array_equal(
            mask,
            [
                [False, True, False, False],
                [True, False, False, False],
                [False, False, False, True],
                [False, False, True, False],
            ],
        )


class BestEpochContractTests(unittest.TestCase):
    def test_improvement_threshold_and_four_epoch_early_stop_are_observable(self):
        selector = BestEpochSelector()

        self.assertTrue(selector.observe(1, 0.5))
        self.assertFalse(selector.observe(2, 0.500_09))
        self.assertFalse(selector.observe(3, 0.500_09))
        self.assertFalse(selector.observe(4, 0.500_09))
        self.assertFalse(selector.should_stop)
        self.assertFalse(selector.observe(5, 0.500_09))
        self.assertTrue(selector.should_stop)
        self.assertEqual(selector.best_epoch, 1)
        self.assertAlmostEqual(selector.best_accuracy, 0.5)


class SelectionAndLedgerContractTests(unittest.TestCase):
    def test_selection_uses_score_then_loss_specific_hyperparameters(self):
        rows = []
        for candidate in candidate_configurations():
            rows.append(
                {
                    "trial_id": candidate.trial_id,
                    "loss": candidate.loss,
                    "hyperparameters_json": json.dumps(candidate.hyperparameters),
                    "seed": candidate.seed,
                    "epochs_attempted": 5,
                    "best_epoch": 3,
                    "best_validation_triplet_accuracy": 0.75,
                    "early_stopped": "false",
                    "images_seen": 1000,
                    "steps_per_epoch": 10,
                    "device": "cuda",
                    "selected": "false",
                    "notes": "",
                }
            )

        selected = select_candidates(rows)

        self.assertEqual(
            [row["trial_id"] for row in selected if row["selected"] == "true"],
            [
                "triplet_margin0p1_seed4002",
                "arcface_m0p2_s30_seed4002",
                "supcon_t0p07_seed4002",
            ],
        )

    def test_writes_complete_contract_ledger_and_local_only_checkpoint_paths(self):
        rows = []
        for candidate in candidate_configurations():
            rows.append(
                {
                    "trial_id": candidate.trial_id,
                    "loss": candidate.loss,
                    "hyperparameters_json": json.dumps(candidate.hyperparameters),
                    "seed": candidate.seed,
                    "epochs_attempted": 2,
                    "best_epoch": 1,
                    "best_validation_triplet_accuracy": 0.5,
                    "early_stopped": "false",
                    "images_seen": 100,
                    "steps_per_epoch": 10,
                    "device": "cuda",
                    "selected": "false",
                    "notes": "contract fixture",
                }
            )
        rows = select_candidates(rows)

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            ledger_path = root / "trial_results.csv"
            write_trial_ledger(ledger_path, rows)
            with ledger_path.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                written = list(reader)

            self.assertEqual(len(written), 10)
            self.assertEqual(
                reader.fieldnames,
                [
                    "trial_id",
                    "loss",
                    "hyperparameters_json",
                    "seed",
                    "epochs_attempted",
                    "best_epoch",
                    "best_validation_triplet_accuracy",
                    "early_stopped",
                    "images_seen",
                    "steps_per_epoch",
                    "device",
                    "selected",
                    "notes",
                ],
            )
            self.assertEqual(
                trial_checkpoint_path(root, rows[0]),
                root / "models" / "metric_learning" / rows[0]["trial_id"] / "best.pt",
            )


if __name__ == "__main__":
    unittest.main()
