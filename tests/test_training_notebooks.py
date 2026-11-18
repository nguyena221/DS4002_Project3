import json
import platform
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch
from unittest import mock

from face_name_metric_learning import collect_environment_record
from face_name_metric_learning import export_baseline_embeddings
from face_name_metric_learning import load_embedding_package
from face_name_metric_learning import load_pretrained_iresnet100
from face_name_metric_learning.handoff import sha256_file
from face_name_metric_learning.iresnet import load_iresnet100


ROOT = Path(__file__).resolve().parents[1]


class NotebookSchemaTests(unittest.TestCase):
    def test_committed_stage_notebooks_are_runnable_and_output_free(self):
        for name in (
            "00_setup_and_smoke_test.ipynb",
            "01_export_baseline_embeddings.ipynb",
            "02_train_metric_models.ipynb",
            "03_export_selected_embeddings.ipynb",
        ):
            path = ROOT / "notebooks" / name
            with self.subTest(notebook=name):
                notebook = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(notebook["nbformat"], 4)
                self.assertEqual(
                    sum("Configuration" in "".join(cell["source"]) for cell in notebook["cells"]),
                    1,
                )
                for cell in notebook["cells"]:
                    if cell["cell_type"] == "code":
                        self.assertEqual(cell["outputs"], [])
                        self.assertIsNone(cell["execution_count"])
                        compile("".join(cell["source"]), str(path), "exec")

    def test_setup_and_baseline_notebooks_use_repository_loader_and_compatible_pins(self):
        for name in (
            "00_setup_and_smoke_test.ipynb",
            "01_export_baseline_embeddings.ipynb",
        ):
            notebook = json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))
            source = "\n".join(
                "".join(cell["source"])
                for cell in notebook["cells"]
                if cell["cell_type"] == "code"
            )

            with self.subTest(notebook=name):
                self.assertIn("face_name_metric_learning.iresnet", source)
                self.assertIn("sys.path.insert", source)
                self.assertIn("Path.cwd().resolve().parents", source)
                self.assertNotIn("pytorch" + "_insightface", source)
                self.assertNotIn("pytorch" + "-insightface", source)

        setup = json.loads(
            (ROOT / "notebooks" / "00_setup_and_smoke_test.ipynb").read_text(
                encoding="utf-8"
            )
        )
        source = "\n".join("".join(cell["source"]) for cell in setup["cells"])
        self.assertIn("torch==2.8.0", source)
        self.assertIn("h5py==3.16.0", source)


class TrainingStagePublicBehaviorTests(unittest.TestCase):
    def test_environment_record_exposes_reproducible_runtime_identity(self):
        record = collect_environment_record()

        self.assertEqual(record["python"], platform.python_version())
        self.assertIn("pytorch", record["packages"])
        self.assertIn("pytorch_metric_learning", record["packages"])
        self.assertIn("h5py", record["packages"])
        self.assertNotIn("pytorch" + "_insightface", record["packages"])
        self.assertIn("available", record["gpu"])
        self.assertIn("device", record)

    def test_default_pretrained_loader_uses_local_implementation_with_strict_state_and_checksum(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            checkpoint_path = Path(directory) / "converted.pth"
            reference = torch.nn.Linear(2, 1)
            torch.save(reference.state_dict(), checkpoint_path)
            checksum = sha256_file(checkpoint_path)

            with mock.patch(
                "face_name_metric_learning.iresnet.build_iresnet100",
                return_value=torch.nn.Linear(2, 1),
            ):
                loaded = load_pretrained_iresnet100(
                    checkpoint_path,
                    expected_sha256=checksum,
                    device="cpu",
                )
                self.assertIsInstance(loaded, torch.nn.Linear)
                self.assertIsInstance(
                    load_iresnet100(
                        checkpoint_path,
                        "cpu",
                        expected_sha256=checksum,
                    ),
                    torch.nn.Linear,
                )
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    load_pretrained_iresnet100(
                        checkpoint_path,
                        expected_sha256="0" * 64,
                        device="cpu",
                    )

                invalid_path = Path(directory) / "invalid.pth"
                torch.save({**reference.state_dict(), "unexpected": torch.ones(1)}, invalid_path)
                with self.assertRaisesRegex(RuntimeError, "Unexpected key"):
                    load_pretrained_iresnet100(invalid_path, device="cpu")

    def test_exports_validation_and_test_with_stable_h5_indices(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.csv"
            hdf5_path = root / "faces.h5"
            output_path = root / "baseline.npz"
            manifest_path.write_text(
                "h5_index,filename,first_name,label,split\n"
                "0,v0.png,Ada,0,validation\n"
                "1,t0.png,Ada,0,test\n"
                "2,v1.png,Bea,1,validation\n",
                encoding="utf-8",
            )
            images = np.zeros((3, 2, 1, 1), dtype=np.float32)
            images[:, 0, 0, 0] = [1.0, 0.0, 0.0]
            images[:, 1, 0, 0] = [0.0, 1.0, 1.0]
            with h5py.File(hdf5_path, "w") as h5:
                h5.create_dataset("images", data=images)
                h5.create_dataset(
                    "split_indices/validation", data=np.array([2, 0], dtype=np.int64)
                )
                h5.create_dataset(
                    "split_indices/test", data=np.array([1], dtype=np.int64)
                )

            class Encoder(torch.nn.Module):
                def forward(self, batch):
                    return batch.flatten(start_dim=1)

            export_baseline_embeddings(
                Encoder(),
                hdf5_path=hdf5_path,
                manifest_path=manifest_path,
                output_path=output_path,
                expected_dimension=2,
                batch_size=1,
                device="cpu",
            )
            package = load_embedding_package(
                output_path, manifest_path=manifest_path, expected_dimension=2
            )

            np.testing.assert_array_equal(
                package["validation_h5_index"], np.array([2, 0], dtype=np.int64)
            )
            np.testing.assert_array_equal(
                package["test_h5_index"], np.array([1], dtype=np.int64)
            )
            np.testing.assert_allclose(
                np.linalg.norm(package["validation_embeddings"], axis=1), 1.0
            )
            np.testing.assert_allclose(
                np.linalg.norm(package["test_embeddings"], axis=1), 1.0
            )


if __name__ == "__main__":
    unittest.main()
