import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from face_name_metric_learning import (
    audit_handoff_package,
    build_handoff_manifest,
    read_handoff_manifest,
    write_handoff_manifest,
)


class HandoffAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.root = Path(self.temp_dir.name)
        self.package_dir = self.root / "artifacts"
        (self.package_dir / "embeddings").mkdir(parents=True)
        self.manifest_path = self.root / "manifest.csv"
        self.hdf5_path = self.root / "faces.h5"
        self.trial_path = self.package_dir / "trial_results.csv"
        self.handoff_path = self.package_dir / "handoff_manifest.json"

        identities = [
            (0, "Ada", 0, "validation"),
            (1, "Ada", 0, "validation"),
            (2, "Bea", 1, "validation"),
            (3, "Bea", 1, "validation"),
            (4, "Ada", 0, "test"),
            (5, "Bea", 1, "test"),
        ]
        with self.manifest_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=["h5_index", "filename", "first_name", "label", "split"],
            )
            writer.writeheader()
            for h5_index, first_name, label, split in identities:
                writer.writerow(
                    {
                        "h5_index": h5_index,
                        "filename": f"{h5_index}.png",
                        "first_name": first_name,
                        "label": label,
                        "split": split,
                    }
                )
        self.hdf5_path.write_bytes(b"synthetic-face-source")

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
        with self.trial_path.open("w", newline="", encoding="utf-8") as stream:
            rows = [
                {
                    "trial_id": f"trial-{index}",
                    "loss": loss,
                    "hyperparameters_json": json.dumps(hyperparameters),
                    "seed": 4002,
                    "epochs_attempted": 2,
                    "best_epoch": 1,
                    "best_validation_triplet_accuracy": 1.0,
                    "early_stopped": "false",
                    "images_seen": 4,
                    "steps_per_epoch": 2,
                    "device": "cpu",
                    "selected": "true" if index in (0, 3, 7) else "false",
                    "notes": "",
                }
                for index, (loss, hyperparameters) in enumerate(candidates)
            ]
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)

        validation_embeddings = np.array(
            [[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float32
        )
        test_embeddings = np.array([[1, 0], [0, 1]], dtype=np.float32)
        for filename in (
            "baseline_iresnet100.npz",
            "triplet_margin_selected.npz",
            "arcface_selected.npz",
            "supcon_selected.npz",
        ):
            np.savez_compressed(
                self.package_dir / "embeddings" / filename,
                validation_h5_index=np.array([2, 0, 3, 1], dtype=np.int64),
                validation_embeddings=np.array(
                    [
                        validation_embeddings[2],
                        validation_embeddings[0],
                        validation_embeddings[3],
                        validation_embeddings[1],
                    ],
                    dtype=np.float32,
                ),
                test_h5_index=np.array([5, 4], dtype=np.int64),
                test_embeddings=np.array(
                    [test_embeddings[1], test_embeddings[0]], dtype=np.float32
                ),
            )

        metadata = build_handoff_manifest(
            self.package_dir,
            manifest_path=self.manifest_path,
            hdf5_path=self.hdf5_path,
            trial_results_path=self.trial_path,
            versions={
                "python": "3.11",
                "pytorch": "2.0",
                "cuda": "12.0",
                "pytorch_metric_learning": "2.0",
                "pytorch_insightface": "0.7.3",
            },
            pretrained_weights={"identifier": "synthetic", "sha256": "fixture"},
            device="cpu",
            child_seed_policy="fixture",
            split_counts={"train": 0, "validation": 4, "test": 2},
            class_count=2,
            expected_dimension=2,
            generated_at="2026-11-18T18:00:00Z",
        )
        self.assertEqual(metadata["audit"], {"status": "not_run"})
        write_handoff_manifest(self.handoff_path, metadata)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_reloads_aligns_and_records_deterministic_metrics_in_manifest(self):
        audited = audit_handoff_package(
            self.package_dir,
            manifest_path=self.manifest_path,
            hdf5_path=self.hdf5_path,
            trial_results_path=self.trial_path,
            handoff_manifest_path=self.handoff_path,
            expected_dimension=2,
            audit_timestamp="2026-11-18T18:05:00Z",
        )

        self.assertEqual(audited["audit"]["status"], "pass")
        self.assertEqual(audited["audit"]["metric"]["sample_id_order"], [0, 1, 2, 3])
        self.assertEqual(
            audited["audit"]["metric"]["models"],
            {
                "baseline_iresnet100": 1.0,
                "triplet_margin_selected": 1.0,
                "arcface_selected": 1.0,
                "supcon_selected": 1.0,
            },
        )
        self.assertEqual(read_handoff_manifest(self.handoff_path), audited)

    def test_records_failure_before_reraising_invalid_package_error(self):
        model_path = (
            self.package_dir / "embeddings" / "baseline_iresnet100.npz"
        )
        np.savez_compressed(
            model_path,
            validation_h5_index=np.array([2, 0, 3, 1], dtype=np.int64),
            validation_embeddings=np.array(
                [[0, 2], [1, 0], [0, 1], [1, 0]], dtype=np.float32
            ),
            test_h5_index=np.array([5, 4], dtype=np.int64),
            test_embeddings=np.array([[0, 1], [1, 0]], dtype=np.float32),
        )

        with self.assertRaisesRegex(ValueError, r"checksums\.embeddings"):
            audit_handoff_package(
                self.package_dir,
                manifest_path=self.manifest_path,
                hdf5_path=self.hdf5_path,
                trial_results_path=self.trial_path,
                handoff_manifest_path=self.handoff_path,
                expected_dimension=2,
                audit_timestamp="2026-11-18T18:06:00Z",
            )

        failed = read_handoff_manifest(self.handoff_path)
        self.assertEqual(failed["audit"]["status"], "fail")
        self.assertEqual(failed["audit"]["timestamp"], "2026-11-18T18:06:00Z")
        self.assertIn("checksums.embeddings", failed["audit"]["error"])


if __name__ == "__main__":
    unittest.main()
