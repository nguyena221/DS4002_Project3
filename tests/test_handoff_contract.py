import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from face_name_metric_learning.handoff import (
    align_embeddings,
    build_handoff_manifest,
    load_embedding_package,
    validate_handoff_manifest,
    validate_trial_ledger,
    validate_manifest_hdf5,
    write_embedding_package,
)


class ManifestHdf5ContractTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.manifest_path = self.root / "manifest.csv"
        self.hdf5_path = self.root / "faces.h5"

        rows = [
            {"h5_index": 0, "filename": "a.png", "first_name": "Ada", "label": 0, "split": "train"},
            {"h5_index": 1, "filename": "b.png", "first_name": "Bea", "label": 1, "split": "validation"},
            {"h5_index": 2, "filename": "c.png", "first_name": "Ada", "label": 0, "split": "test"},
        ]
        with self.manifest_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)

        with h5py.File(self.hdf5_path, "w") as h5:
            h5.attrs["complete"] = True
            h5.create_dataset("filenames", data=np.array(["a.png", "b.png", "c.png"], dtype="S8"))
            h5.create_dataset("labels", data=np.array([0, 1, 0], dtype=np.int64))
            h5.create_dataset("split", data=np.array([0, 1, 2], dtype=np.int8))
            h5.create_dataset("class_names", data=np.array(["Ada", "Bea"], dtype=h5py.string_dtype()))
            split_group = h5.create_group("split_indices")
            split_group.create_dataset("train", data=np.array([0], dtype=np.int64))
            split_group.create_dataset("validation", data=np.array([1], dtype=np.int64))
            split_group.create_dataset("test", data=np.array([2], dtype=np.int64))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_validates_manifest_identity_and_split_coverage(self):
        result = validate_manifest_hdf5(
            self.manifest_path,
            self.hdf5_path,
            expected_split_counts={"train": 1, "validation": 1, "test": 1},
            expected_class_count=2,
        )

        self.assertEqual(result["split_counts"], {"train": 1, "validation": 1, "test": 1})
        self.assertEqual(result["class_count"], 2)

    def test_rejects_an_hdf5_identity_not_matching_the_manifest(self):
        with h5py.File(self.hdf5_path, "r+") as h5:
            h5["split_indices/validation"][0] = 0

        with self.assertRaisesRegex(ValueError, r"split_indices\.validation"):
            validate_manifest_hdf5(
                self.manifest_path,
                self.hdf5_path,
                expected_split_counts={"train": 1, "validation": 1, "test": 1},
                expected_class_count=2,
            )


class EmbeddingPackageContractTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.manifest_path = self.root / "manifest.csv"
        with self.manifest_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=["h5_index", "filename", "first_name", "label", "split"],
            )
            writer.writeheader()
            writer.writerows(
                [
                    {"h5_index": 10, "filename": "v2.png", "first_name": "Ada", "label": 0, "split": "validation"},
                    {"h5_index": 11, "filename": "t.png", "first_name": "Ada", "label": 0, "split": "test"},
                    {"h5_index": 12, "filename": "v1.png", "first_name": "Bea", "label": 1, "split": "validation"},
                ]
            )
        self.package_path = self.root / "model.npz"
        self.validation_indices = np.array([12, 10], dtype=np.int64)
        self.validation_embeddings = np.array([[1, 0], [0, 1]], dtype=np.float32)
        self.test_indices = np.array([11], dtype=np.int64)
        self.test_embeddings = np.array([[0, -1]], dtype=np.float32)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write(self):
        return write_embedding_package(
            self.package_path,
            self.validation_indices,
            self.validation_embeddings,
            self.test_indices,
            self.test_embeddings,
            manifest_path=self.manifest_path,
            expected_dimension=2,
        )

    def test_round_trips_only_required_keys_and_aligns_by_h5_index(self):
        self._write()

        package = load_embedding_package(
            self.package_path, manifest_path=self.manifest_path, expected_dimension=2
        )
        aligned = align_embeddings(
            package["validation_h5_index"],
            package["validation_embeddings"],
            np.array([10, 12], dtype=np.int64),
        )

        self.assertEqual(
            set(np.load(self.package_path).files),
            {
                "validation_h5_index",
                "validation_embeddings",
                "test_h5_index",
                "test_embeddings",
            },
        )
        np.testing.assert_array_equal(aligned, [[0, 1], [1, 0]])

    def test_rejects_non_float32_non_finite_and_non_unit_embeddings(self):
        cases = [
            ("dtype", self.validation_embeddings.astype(np.float64), "validation_embeddings.dtype"),
            ("finite", np.array([[np.inf, 0], [0, 1]], dtype=np.float32), "validation_embeddings.finite"),
            ("norm", np.array([[2, 0], [0, 1]], dtype=np.float32), "validation_embeddings.norm"),
        ]
        for name, embeddings, message in cases:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, message):
                write_embedding_package(
                    self.package_path,
                    self.validation_indices,
                    embeddings,
                    self.test_indices,
                    self.test_embeddings,
                    manifest_path=self.manifest_path,
                    expected_dimension=2,
                )

    def test_rejects_duplicate_or_wrong_split_indices(self):
        with self.assertRaisesRegex(ValueError, r"validation_h5_index\.unique"):
            write_embedding_package(
                self.package_path,
                np.array([10, 10], dtype=np.int64),
                self.validation_embeddings,
                self.test_indices,
                self.test_embeddings,
                manifest_path=self.manifest_path,
                expected_dimension=2,
            )

        with self.assertRaisesRegex(ValueError, r"validation_h5_index\.coverage"):
            write_embedding_package(
                self.package_path,
                np.array([11, 10], dtype=np.int64),
                self.validation_embeddings,
                self.test_indices,
                self.test_embeddings,
                manifest_path=self.manifest_path,
                expected_dimension=2,
            )


class TrialLedgerContractTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = Path(self.temp_dir.name) / "trial_results.csv"

    def tearDown(self):
        self.temp_dir.cleanup()

    @staticmethod
    def rows():
        candidates = [
            ("triplet_margin", {"margin": 0.1}),
            ("triplet_margin", {"margin": 0.2}),
            ("triplet_margin", {"margin": 0.4}),
            ("arcface", {"margin": 0.2, "scale": 30}),
            ("arcface", {"margin": 0.2, "scale": 64}),
            ("arcface", {"margin": 0.4, "scale": 30}),
            ("arcface", {"margin": 0.4, "scale": 64}),
            ("supcon", {"temperature": 0.07}),
            ("supcon", {"temperature": 0.1}),
            ("supcon", {"temperature": 0.2}),
        ]
        return [
            {
                "trial_id": f"trial-{index}",
                "loss": loss,
                "hyperparameters_json": json.dumps(hyperparameters),
                "seed": 4002,
                "epochs_attempted": 3,
                "best_epoch": 2,
                "best_validation_triplet_accuracy": 0.5,
                "early_stopped": "false",
                "images_seen": 100,
                "steps_per_epoch": 10,
                "device": "cuda",
                "selected": "true" if index in (0, 3, 7) else "false",
                "notes": "",
            }
            for index, (loss, hyperparameters) in enumerate(candidates)
        ]

    def write_rows(self, rows):
        with self.path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)

    def test_validates_ten_candidate_grid_and_one_selection_per_loss(self):
        self.write_rows(self.rows())

        ledger = validate_trial_ledger(self.path)

        self.assertEqual(len(ledger["rows"]), 10)
        self.assertEqual(
            ledger["selected_trials"],
            {"triplet_margin": "trial-0", "arcface": "trial-3", "supcon": "trial-7"},
        )

    def test_rejects_missing_or_duplicated_selected_trial(self):
        rows = self.rows()
        rows[1]["selected"] = "true"
        self.write_rows(rows)

        with self.assertRaisesRegex(ValueError, r"trial_results\.triplet_margin\.selected"):
            validate_trial_ledger(self.path)

    def test_rejects_incomplete_candidate_grid(self):
        rows = self.rows()
        del rows[-1]
        self.write_rows(rows)

        with self.assertRaisesRegex(ValueError, r"trial_results\.count"):
            validate_trial_ledger(self.path)


class HandoffManifestContractTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.package_dir = self.root / "artifacts"
        (self.package_dir / "embeddings").mkdir(parents=True)
        self.manifest_path = self.root / "manifest.csv"
        self.hdf5_path = self.root / "faces.h5"
        self.trial_path = self.package_dir / "trial_results.csv"
        self.versions = {
            "python": "3.11",
            "pytorch": "2.0",
            "cuda": "12.0",
            "pytorch_metric_learning": "2.0",
            "pytorch_insightface": "0.7.3",
        }

        rows = [
            {"h5_index": 0, "filename": "v.png", "first_name": "Ada", "label": 0, "split": "validation"},
            {"h5_index": 1, "filename": "t.png", "first_name": "Ada", "label": 0, "split": "test"},
        ]
        with self.manifest_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
        self.hdf5_path.write_bytes(b"faces")
        trial_rows = TrialLedgerContractTests.rows()
        with self.trial_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=trial_rows[0])
            writer.writeheader()
            writer.writerows(trial_rows)
        for filename in (
            "baseline_iresnet100.npz",
            "triplet_margin_selected.npz",
            "arcface_selected.npz",
            "supcon_selected.npz",
        ):
            np.savez_compressed(
                self.package_dir / "embeddings" / filename,
                validation_h5_index=np.array([0], dtype=np.int64),
                validation_embeddings=np.array([[1, 0]], dtype=np.float32),
                test_h5_index=np.array([1], dtype=np.int64),
                test_embeddings=np.array([[0, 1]], dtype=np.float32),
            )

    def tearDown(self):
        self.temp_dir.cleanup()

    def build(self):
        return build_handoff_manifest(
            self.package_dir,
            manifest_path=self.manifest_path,
            hdf5_path=self.hdf5_path,
            trial_results_path=self.trial_path,
            versions=self.versions,
            pretrained_weights={"identifier": "insightface-iresnet100", "sha256": "weight-checksum"},
            device="cuda:0",
            child_seed_policy="named deterministic streams",
            expected_dimension=2,
            split_counts={"train": 0, "validation": 1, "test": 1},
            class_count=1,
            generated_at="2026-11-18T18:00:00Z",
            git_commit="abc123",
            audit={"status": "pass", "timestamp": "2026-11-18T18:01:00Z"},
        )

    def test_builds_schema_v1_checksums_selections_and_metric_definitions(self):
        metadata = self.build()

        self.assertEqual(metadata["schema_version"], 1)
        self.assertEqual(metadata["selected_trials"]["arcface"], "trial-3")
        self.assertEqual(metadata["checksums"]["manifest.csv"], hashlib.sha256(self.manifest_path.read_bytes()).hexdigest())
        self.assertIn("strictly greater", metadata["metric_definitions"]["triplet_accuracy"])
        self.assertIs(validate_handoff_manifest(
            metadata,
            self.package_dir,
            manifest_path=self.manifest_path,
            hdf5_path=self.hdf5_path,
            trial_results_path=self.trial_path,
            expected_dimension=2,
        ), metadata)

    def test_rejects_a_checksum_or_selected_id_that_does_not_match_inputs(self):
        metadata = self.build()
        metadata["checksums"]["faces.h5"] = "0" * 64
        with self.assertRaisesRegex(ValueError, r"checksums\.faces\.h5"):
            validate_handoff_manifest(
                metadata,
                self.package_dir,
                manifest_path=self.manifest_path,
                hdf5_path=self.hdf5_path,
                trial_results_path=self.trial_path,
                expected_dimension=2,
            )

        metadata = self.build()
        metadata["selected_trials"]["supcon"] = "wrong"
        with self.assertRaisesRegex(ValueError, r"selected_trials\.supcon"):
            validate_handoff_manifest(
                metadata,
                self.package_dir,
                manifest_path=self.manifest_path,
                hdf5_path=self.hdf5_path,
                trial_results_path=self.trial_path,
                expected_dimension=2,
            )


if __name__ == "__main__":
    unittest.main()
