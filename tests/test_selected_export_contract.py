import csv
import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

from face_name_metric_learning import (
    build_handoff_manifest_from_environment,
    export_selected_embeddings,
    load_embedding_package,
    read_selected_trials,
)


class SelectedExportContractTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.root = Path(self.temp_dir.name)
        self.manifest_path = self.root / "data" / "processed" / "manifest.csv"
        self.hdf5_path = self.root / "data" / "processed" / "faces.h5"
        self.trial_path = (
            self.root
            / "artifacts"
            / "face_name_metric_learning"
            / "trial_results.csv"
        )
        self.package_dir = self.trial_path.parent
        self.manifest_path.parent.mkdir(parents=True)
        self.package_dir.mkdir(parents=True)
        with self.manifest_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=["h5_index", "filename", "first_name", "label", "split"],
            )
            writer.writeheader()
            writer.writerows(
                [
                    {
                        "h5_index": 0,
                        "filename": "v.png",
                        "first_name": "Ada",
                        "label": 0,
                        "split": "validation",
                    },
                    {
                        "h5_index": 1,
                        "filename": "t.png",
                        "first_name": "Ada",
                        "label": 0,
                        "split": "test",
                    },
                ]
            )
        images = np.array(
            [[[1.0], [0.0]], [[0.0], [1.0]]], dtype=np.float32
        )
        with h5py.File(self.hdf5_path, "w") as h5:
            h5.create_dataset("images", data=images[:, :, :, None])
            h5.create_dataset(
                "split_indices/validation", data=np.array([0], dtype=np.int64)
            )
            h5.create_dataset(
                "split_indices/test", data=np.array([1], dtype=np.int64)
            )

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
        rows = []
        for index, (loss, hyperparameters) in enumerate(candidates):
            trial_id = f"trial-{index}"
            rows.append(
                {
                    "trial_id": trial_id,
                    "loss": loss,
                    "hyperparameters_json": json.dumps(hyperparameters),
                    "seed": 4002,
                    "epochs_attempted": 2,
                    "best_epoch": 1,
                    "best_validation_triplet_accuracy": 0.7,
                    "early_stopped": "false",
                    "images_seen": 2,
                    "steps_per_epoch": 1,
                    "device": "cuda:0",
                    "selected": "true" if index in (0, 3, 7) else "false",
                    "notes": "",
                }
            )
        with self.trial_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_reads_exactly_one_selected_row_for_each_supported_loss(self):
        selected = read_selected_trials(self.trial_path)

        self.assertEqual(
            set(selected),
            {"triplet_margin", "arcface", "supcon"},
        )
        self.assertEqual(
            [selected[loss]["trial_id"] for loss in selected],
            ["trial-0", "trial-3", "trial-7"],
        )

    def test_exports_stable_selected_npzs_through_public_model_seam(self):
        loaded_paths = []
        for trial in read_selected_trials(self.trial_path).values():
            checkpoint = (
                self.root
                / "models"
                / "metric_learning"
                / trial["trial_id"]
                / "best.pt"
            )
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"synthetic-checkpoint")

        class Encoder(torch.nn.Module):
            def forward(self, batch):
                return batch.flatten(start_dim=1)

        def load_checkpoint(path, device):
            loaded_paths.append((Path(path), device))
            return Encoder()

        outputs = export_selected_embeddings(
            repository_root=self.root,
            manifest_path=self.manifest_path,
            hdf5_path=self.hdf5_path,
            trial_results_path=self.trial_path,
            package_dir=self.package_dir,
            model_loader=load_checkpoint,
            batch_size=1,
            device="cpu",
            expected_dimension=2,
        )

        self.assertEqual(
            list(outputs),
            [
                "triplet_margin_selected.npz",
                "arcface_selected.npz",
                "supcon_selected.npz",
            ],
        )
        self.assertEqual(len(loaded_paths), 3)
        for output in outputs.values():
            self.assertFalse(output.name == "best.pt")
            package = load_embedding_package(
                output, manifest_path=self.manifest_path, expected_dimension=2
            )
            self.assertEqual(
                set(package),
                {
                    "validation_h5_index",
                    "validation_embeddings",
                    "test_h5_index",
                    "test_embeddings",
                },
            )
        self.assertEqual(
            {path.relative_to(self.root).as_posix() for path, _ in loaded_paths},
            {
                "models/metric_learning/trial-0/best.pt",
                "models/metric_learning/trial-3/best.pt",
                "models/metric_learning/trial-7/best.pt",
            },
        )

    def test_missing_selected_checkpoint_fails_before_any_export(self):
        with self.assertRaisesRegex(FileNotFoundError, r"models.*trial-0.*best\.pt"):
            export_selected_embeddings(
                repository_root=self.root,
                manifest_path=self.manifest_path,
                hdf5_path=self.hdf5_path,
                trial_results_path=self.trial_path,
                package_dir=self.package_dir,
                model_loader=lambda *_: self.fail("loader should not run"),
                device="cpu",
                expected_dimension=2,
            )

    def test_builds_manifest_from_persisted_environment_and_training_ledger(self):
        selected = read_selected_trials(self.trial_path)
        for trial in selected.values():
            checkpoint = (
                self.root
                / "models"
                / "metric_learning"
                / trial["trial_id"]
                / "best.pt"
            )
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"synthetic-checkpoint")

        class Encoder(torch.nn.Module):
            def forward(self, batch):
                return batch.flatten(start_dim=1)

        export_selected_embeddings(
            repository_root=self.root,
            manifest_path=self.manifest_path,
            hdf5_path=self.hdf5_path,
            trial_results_path=self.trial_path,
            package_dir=self.package_dir,
            model_loader=lambda *_: Encoder(),
            batch_size=1,
            device="cpu",
            expected_dimension=2,
        )
        np.savez_compressed(
            self.package_dir / "embeddings" / "baseline_iresnet100.npz",
            validation_h5_index=np.array([0], dtype=np.int64),
            validation_embeddings=np.array([[1, 0]], dtype=np.float32),
            test_h5_index=np.array([1], dtype=np.int64),
            test_embeddings=np.array([[0, 1]], dtype=np.float32),
        )
        environment_path = self.package_dir / "setup_environment.json"
        environment_path.write_text(
            json.dumps(
                {
                    "python": "3.11.9",
                    "packages": {
                        "pytorch": "2.0",
                        "pytorch_metric_learning": "2.0",
                        "h5py": "3.16.0",
                    },
                    "cuda": "12.0",
                    "device": "cuda:0",
                    "pretrained_weights": {
                        "identifier": "insightface-iresnet100",
                        "sha256": "approved-weight-checksum",
                    },
                }
            ),
            encoding="utf-8",
        )

        metadata = build_handoff_manifest_from_environment(
            self.package_dir,
            environment_record_path=environment_path,
            manifest_path=self.manifest_path,
            hdf5_path=self.hdf5_path,
            trial_results_path=self.trial_path,
            child_seed_policy="SHA-256 named deterministic streams",
            expected_dimension=2,
            split_counts={"train": 0, "validation": 1, "test": 1},
            class_count=1,
            generated_at="2026-11-18T18:00:00Z",
            git_commit="abc123",
        )

        self.assertEqual(metadata["selected_trials"]["triplet_margin"], "trial-0")
        self.assertEqual(metadata["versions"]["pytorch"], "2.0")
        self.assertEqual(metadata["device"], "cuda:0")
        self.assertEqual(metadata["audit"], {"status": "not_run"})
        self.assertFalse(
            any(
                path.suffix == ".pt"
                for path in self.package_dir.rglob("*")
                if path.is_file()
            )
        )


if __name__ == "__main__":
    unittest.main()
