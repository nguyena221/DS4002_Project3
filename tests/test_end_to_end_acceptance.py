import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from face_name_metric_learning import (
    build_handoff_manifest,
    evaluate_handoff_package,
    write_handoff_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
LOSSES = (
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
)


class EndToEndAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.source = Path(self.temp_dir.name) / "source"
        self.package_dir = self.source / "artifacts" / "face_name_metric_learning"
        self.manifest_path = self.source / "data" / "processed" / "manifest.csv"
        self.hdf5_path = self.source / "data" / "processed" / "faces.h5"
        self.package_dir.mkdir(parents=True)
        self.manifest_path.parent.mkdir(parents=True)
        (self.package_dir / "embeddings").mkdir()

        rows = []
        for split, start in (("validation", 0), ("test", 12)):
            for offset, (name, label) in enumerate(
                (("Ada", 0), ("Bea", 1), ("Ada", 0), ("Bea", 1))
                * 3
            ):
                index = start + offset
                rows.append(
                    {
                        "h5_index": index,
                        "filename": f"{index}.png",
                        "first_name": name,
                        "label": label,
                        "split": split,
                    }
                )
        with self.manifest_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
        self.hdf5_path.write_bytes(b"tiny source fingerprint")

        trials = []
        for index, (loss, hyperparameters) in enumerate(LOSSES):
            trials.append(
                {
                    "trial_id": f"trial-{index}",
                    "loss": loss,
                    "hyperparameters_json": json.dumps(hyperparameters),
                    "seed": 4002,
                    "epochs_attempted": 2,
                    "best_epoch": 1,
                    "best_validation_triplet_accuracy": 0.5 + index / 100,
                    "early_stopped": "false",
                    "images_seen": 24,
                    "steps_per_epoch": 12,
                    "device": "cpu",
                    "selected": "true" if index in (0, 3, 7) else "false",
                    "notes": "acceptance fixture",
                }
            )
        trial_path = self.package_dir / "trial_results.csv"
        with trial_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=trials[0])
            writer.writeheader()
            writer.writerows(trials)

        rng = np.random.default_rng(17)
        for model_number, filename in enumerate(
            (
                "baseline_iresnet100.npz",
                "triplet_margin_selected.npz",
                "arcface_selected.npz",
                "supcon_selected.npz",
            )
        ):
            package = {}
            for split, start in (("validation", 0), ("test", 12)):
                indices = np.arange(start, start + 12, dtype=np.int64)
                angles = np.where(
                    (indices - start) % 2 == 0, 0.0, np.pi / 2
                )
                scale = 0.7 / (model_number + 1)
                angles = angles + rng.normal(0, scale, 12)
                vectors = np.column_stack(
                    [
                        np.cos(angles),
                        np.sin(angles),
                    ]
                ).astype(np.float32)
                order = rng.permutation(12)
                package[f"{split}_h5_index"] = indices[order]
                package[f"{split}_embeddings"] = vectors[order]
            np.savez_compressed(
                self.package_dir / "embeddings" / filename, **package
            )

        metadata = build_handoff_manifest(
            self.package_dir,
            manifest_path=self.manifest_path,
            hdf5_path=self.hdf5_path,
            trial_results_path=trial_path,
            versions={
                "python": "3.11",
                "pytorch": "2.0",
                "cuda": "12.0",
                "pytorch_metric_learning": "2.0",
                "pytorch_insightface": "0.7.3",
            },
            pretrained_weights={"identifier": "fixture", "sha256": "fixture"},
            device="cpu",
            child_seed_policy="fixture named streams",
            split_counts={"train": 0, "validation": 12, "test": 12},
            class_count=2,
            expected_dimension=2,
            generated_at="2026-11-18T18:00:00Z",
            git_commit="acceptance-fixture",
            audit={"status": "pass", "timestamp": "2026-11-18T18:01:00Z"},
        )
        write_handoff_manifest(
            self.package_dir / "handoff_manifest.json", metadata
        )
        readme_source = ROOT / "artifacts" / "face_name_metric_learning" / "README.md"
        shutil.copy2(readme_source, self.package_dir / "README.md")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_fresh_evaluator_consumes_only_copied_package_and_shared_manifest(self):
        consumer = Path(self.temp_dir.name) / "fresh-colab"
        copied_package = (
            consumer / "artifacts" / "face_name_metric_learning"
        )
        copied_manifest = consumer / "data" / "processed" / "manifest.csv"
        copied_package.mkdir(parents=True)
        copied_manifest.parent.mkdir(parents=True)
        shutil.copytree(self.package_dir, copied_package, dirs_exist_ok=True)
        shutil.copy2(self.manifest_path, copied_manifest)

        result = evaluate_handoff_package(
            copied_package,
            manifest_path=copied_manifest,
            expected_dimension=2,
            block_size=3,
            bootstrap_replicates=10,
            direct_oracle=True,
        )

        self.assertEqual(
            list(result["models"]),
            [
                "baseline_iresnet100",
                "triplet_margin_selected",
                "arcface_selected",
                "supcon_selected",
            ],
        )
        for model in result["models"].values():
            self.assertEqual(set(model["splits"]), {"validation", "test"})
            for split in model["splits"].values():
                self.assertEqual(split["blocked"], split["direct"])
        self.assertEqual(
            set(result["bootstrap"]),
            {"triplet_margin_selected", "arcface_selected", "supcon_selected"},
        )
        for split_intervals in result["bootstrap"].values():
            self.assertEqual(set(split_intervals), {"validation", "test"})
            for interval in split_intervals.values():
                self.assertEqual(len(interval["confidence_interval"]), 2)
                self.assertEqual(len(interval["similarity_lifts"]), 10)
                self.assertEqual(
                    len(interval["baseline_triplet_accuracies"]), 10
                )
                self.assertEqual(
                    len(interval["comparison_triplet_accuracies"]), 10
                )
        self.assertNotIn("faces.h5", [path.name for path in consumer.rglob("*")])

    def test_documented_handoff_and_notebook_paths_exist_and_match(self):
        readme = (
            ROOT / "artifacts" / "face_name_metric_learning" / "README.md"
        ).read_text(encoding="utf-8")
        for relative in (
            "trial_results.csv",
            "handoff_manifest.json",
            "embeddings/baseline_iresnet100.npz",
            "embeddings/triplet_margin_selected.npz",
            "embeddings/arcface_selected.npz",
            "embeddings/supcon_selected.npz",
            "data/processed/manifest.csv",
        ):
            self.assertIn(relative, readme)

        design = (
            ROOT / "docs" / "face-name-metric-learning-handoff.md"
        ).read_text(encoding="utf-8")
        for number in range(4):
            notebook_name = (
                "00_setup_and_smoke_test.ipynb",
                "01_export_baseline_embeddings.ipynb",
                "02_train_metric_models.ipynb",
                "03_export_selected_embeddings.ipynb",
            )[number]
            notebook_path = ROOT / "notebooks" / notebook_name
            self.assertTrue(notebook_path.is_file())
            self.assertIn(f"notebooks/{notebook_name}", design)


if __name__ == "__main__":
    unittest.main()
