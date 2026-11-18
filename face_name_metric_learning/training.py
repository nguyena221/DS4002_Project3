"""Runtime and embedding-export helpers shared by training notebooks."""

import importlib
import inspect
import json
import platform
from importlib import metadata
from pathlib import Path

import h5py
import numpy as np

from .handoff import EMBEDDING_DIMENSION, sha256_file
from .handoff import write_embedding_package


PACKAGE_DISTRIBUTIONS = {
    "pytorch": "torch",
    "pytorch_metric_learning": "pytorch-metric-learning",
    "pytorch_insightface": "pytorch-insightface",
}


def _distribution_version(distribution_name):
    try:
        return metadata.version(distribution_name)
    except metadata.PackageNotFoundError:
        return None


def collect_environment_record(*, torch_module=None):
    """Capture package, Python, device, and GPU facts without downloading models."""
    if torch_module is None:
        torch_module = importlib.import_module("torch")
    cuda_available = bool(torch_module.cuda.is_available())
    gpu_names = (
        [
            torch_module.cuda.get_device_name(index)
            for index in range(torch_module.cuda.device_count())
        ]
        if cuda_available
        else []
    )
    device = "cuda:0" if cuda_available else "cpu"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            public_name: _distribution_version(distribution_name)
            for public_name, distribution_name in PACKAGE_DISTRIBUTIONS.items()
        },
        "cuda": torch_module.version.cuda,
        "cudnn": (
            torch_module.backends.cudnn.version()
            if hasattr(torch_module.backends.cudnn, "version")
            and torch_module.backends.cudnn.is_available()
            else None
        ),
        "device": device,
        "gpu": {
            "available": cuda_available,
            "count": torch_module.cuda.device_count(),
            "names": gpu_names,
        },
    }


def write_environment_record(path, record):
    """Write an environment record for later handoff-manifest construction."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def _load_state_dict(model, checkpoint):
    if hasattr(checkpoint, "keys") and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if hasattr(checkpoint, "keys"):
        checkpoint = {key.removeprefix("module."): value for key, value in checkpoint.items()}
    model.load_state_dict(checkpoint, strict=True)


def load_pretrained_iresnet100(
    pretrained_weights_path=None,
    *,
    module_name="pytorch_insightface.backbones.iresnet",
    pretrained=True,
    device=None,
    expected_sha256=None,
):
    """Load a trainable PyTorch IResNet-100 from the configured implementation."""
    torch = importlib.import_module("torch")
    backbone_module = importlib.import_module(module_name)
    factory = getattr(backbone_module, "iresnet100")
    signature = inspect.signature(factory)
    kwargs = {}
    if "pretrained" in signature.parameters and pretrained_weights_path is None:
        kwargs["pretrained"] = pretrained
    model = factory(**kwargs)
    if pretrained_weights_path is not None:
        path = Path(pretrained_weights_path)
        if not path.is_file():
            raise FileNotFoundError(f"pretrained_weights_path: {path} does not exist")
        if expected_sha256:
            actual_sha256 = sha256_file(path)
            if actual_sha256 != expected_sha256:
                raise ValueError("pretrained_weights.sha256: checksum mismatch")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        _load_state_dict(model, checkpoint)
    model.eval()
    if device is not None:
        model.to(device)
    return model


def _embed_indices(model, images_dataset, indices, *, batch_size, device, torch):
    if batch_size <= 0:
        raise ValueError("batch_size: must be positive")
    result = []
    for start in range(0, len(indices), batch_size):
        batch_indices = np.asarray(indices[start : start + batch_size], dtype=np.int64)
        sorted_order = np.argsort(batch_indices)
        restored_order = np.argsort(sorted_order)
        loaded = images_dataset[batch_indices[sorted_order]]
        batch = torch.from_numpy(np.asarray(loaded)[restored_order]).to(device)
        output = model(batch)
        if not hasattr(output, "detach"):
            output = torch.as_tensor(output)
        output = output.detach().to(dtype=torch.float32, device="cpu")
        if output.ndim != 2:
            raise ValueError(f"model output: expected 2-D embeddings, got {tuple(output.shape)}")
        result.append(output)
    return torch.cat(result, dim=0).numpy().astype(np.float32, copy=False)


def _normalize_rows(embeddings):
    embeddings = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms == 0):
        raise ValueError("embeddings: all rows must be finite and nonzero")
    return (embeddings / norms).astype(np.float32, copy=False)


def export_baseline_embeddings(
    model,
    *,
    hdf5_path,
    manifest_path,
    output_path,
    batch_size=128,
    device="cuda:0",
    expected_dimension=EMBEDDING_DIMENSION,
):
    """Export augmentation-free validation/test vectors using stable HDF5 indices."""
    torch = importlib.import_module("torch")
    model = model.to(device)
    model.eval()
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("device: CUDA was requested but is unavailable")
    with h5py.File(hdf5_path, "r") as h5:
        required = {"images", "split_indices/validation", "split_indices/test"}
        missing = {key for key in required if key not in h5}
        if missing:
            raise ValueError(f"faces.h5: missing required datasets {sorted(missing)}")
        packages = {}
        with torch.inference_mode():
            for split in ("validation", "test"):
                indices = np.asarray(
                    h5[f"split_indices/{split}"][:], dtype=np.int64
                )
                embeddings = _embed_indices(
                    model,
                    h5["images"],
                    indices,
                    batch_size=batch_size,
                    device=device,
                    torch=torch,
                )
                if embeddings.shape[1] != expected_dimension:
                    raise ValueError(
                        f"{split}_embeddings.shape: expected (_, {expected_dimension}), "
                        f"got {embeddings.shape}"
                    )
                packages[f"{split}_h5_index"] = indices
                packages[f"{split}_embeddings"] = _normalize_rows(embeddings)

    path = write_embedding_package(
        output_path,
        packages["validation_h5_index"],
        packages["validation_embeddings"],
        packages["test_h5_index"],
        packages["test_embeddings"],
        manifest_path=manifest_path,
        expected_dimension=expected_dimension,
    )
    return path
