"""Reusable configuration, sampling, selection, and ledger behavior for training."""

import csv
import hashlib
import json
import importlib
import inspect
import platform
from importlib import metadata
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import h5py

from .handoff import EMBEDDING_DIMENSION, sha256_file, write_embedding_package


LOSSES = ("triplet_margin", "arcface", "supcon")
TRIAL_LEDGER_FIELDS = (
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
)


@dataclass(frozen=True)
class TrainingConfig:
    root_seed: int = 4002
    learning_rate: float = 1e-5
    betas: tuple = (0.9, 0.999)
    weight_decay: float = 0.0
    max_epochs: int = 15
    early_stop_patience: int = 4
    minimum_improvement: float = 1e-4
    flip_probability: float = 0.5
    validation_block_size: int = 1024
    embedding_dimension: int = 512
    num_classes: int = 100


DEFAULT_TRAINING_CONFIG = TrainingConfig()


@dataclass(frozen=True)
class NameBalancedBatchPlan:
    names_per_batch: int
    images_per_name: int

    def __post_init__(self):
        if self.names_per_batch <= 0 or self.images_per_name <= 0:
            raise ValueError("names_per_batch and images_per_name must be positive")
        if self.names_per_batch < 2:
            raise ValueError("names_per_batch must permit different-name negatives")
        if self.images_per_name < 2:
            raise ValueError("images_per_name must permit same-name positives")

    @property
    def batch_size(self):
        return self.names_per_batch * self.images_per_name

    @classmethod
    def names_16_by_8(cls):
        return cls(16, 8)

    @classmethod
    def names_8_by_8(cls):
        return cls(8, 8)

    @classmethod
    def steps_for_images(cls, image_count, plan):
        if image_count < plan.batch_size:
            return 0
        return int(image_count) // plan.batch_size

    def __str__(self):
        return f"{self.names_per_batch}x{self.images_per_name}"


@dataclass(frozen=True)
class CandidateConfiguration:
    loss: str
    hyperparameters: dict
    seed: int = DEFAULT_TRAINING_CONFIG.root_seed

    @property
    def trial_id(self):
        try:
            suffix = _LOSS_TRIAL_SUFFIXES[self.loss](self.hyperparameters)
        except KeyError as error:
            raise ValueError(f"unknown loss {self.loss!r}") from error
        return f"{self.loss}_{suffix}_seed{self.seed}".replace(
            "triplet_margin_margin", "triplet_margin"
        )


class BestEpochSelector:
    """Track strict-threshold validation improvements and bounded early stopping."""

    def __init__(
        self,
        *,
        patience=DEFAULT_TRAINING_CONFIG.early_stop_patience,
        minimum_improvement=DEFAULT_TRAINING_CONFIG.minimum_improvement,
    ):
        if patience <= 0:
            raise ValueError("patience must be positive")
        if minimum_improvement < 0:
            raise ValueError("minimum_improvement cannot be negative")
        self.patience = int(patience)
        self.minimum_improvement = float(minimum_improvement)
        self.best_accuracy = -np.inf
        self.best_epoch = 0
        self.stale_epochs = 0

    def observe(self, epoch, accuracy):
        accuracy = float(accuracy)
        if not np.isfinite(accuracy) or not 0.0 <= accuracy <= 1.0:
            raise ValueError("accuracy must be finite and between 0 and 1")
        improved = accuracy >= self.best_accuracy + self.minimum_improvement
        if improved:
            self.best_accuracy = accuracy
            self.best_epoch = int(epoch)
            self.stale_epochs = 0
        else:
            self.stale_epochs += 1
        return improved

    @property
    def should_stop(self):
        return self.stale_epochs >= self.patience


def _id_number(value):
    return format(value, "g").replace(".", "p")


_LOSS_TRIAL_SUFFIXES = {
    "triplet_margin": lambda hyperparameters: (
        f"margin{_id_number(hyperparameters['margin'])}"
    ),
    "arcface": lambda hyperparameters: (
        f"m{_id_number(hyperparameters['margin'])}"
        f"_s{hyperparameters['scale']}"
    ),
    "supcon": lambda hyperparameters: (
        f"t{_id_number(hyperparameters['temperature'])}"
    ),
}


def candidate_configurations(seed=DEFAULT_TRAINING_CONFIG.root_seed):
    """Return the agreed 3/4/3 grid in deterministic order."""
    return [
        *(CandidateConfiguration("triplet_margin", {"margin": value}, seed) for value in (0.1, 0.2, 0.4)),
        *(
            CandidateConfiguration("arcface", {"margin": margin, "scale": scale}, seed)
            for margin in (0.2, 0.4)
            for scale in (30, 64)
        ),
        *(CandidateConfiguration("supcon", {"temperature": value}, seed) for value in (0.07, 0.10, 0.20)),
    ]


def named_child_seed(root_seed, concern, scope):
    """Derive a stable 32-bit seed for an independently named RNG stream."""
    payload = json.dumps(
        {
            "root_seed": int(root_seed),
            "concern": str(concern),
            "scope": str(scope),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def child_seed(candidate, concern):
    """Derive a named seed scoped to one candidate configuration."""
    return named_child_seed(candidate.seed, concern, candidate.trial_id)


def shuffled_name_balanced_batches(h5_index, labels, plan, rng):
    """Yield full P-by-K batches, sampling distinct source images within a batch."""
    h5_index = np.asarray(h5_index, dtype=np.int64)
    labels = np.asarray(labels)
    if h5_index.ndim != 1 or labels.ndim != 1 or len(h5_index) != len(labels):
        raise ValueError("h5_index and labels must be same-length one-dimensional arrays")
    if len(np.unique(h5_index)) != len(h5_index):
        raise ValueError("h5_index must contain unique source identities")

    classes = np.unique(labels)
    if len(classes) < plan.names_per_batch:
        raise ValueError("not enough names for names_per_batch")
    by_class = {label: np.flatnonzero(labels == label) for label in classes}
    if min(map(len, by_class.values())) < plan.images_per_name:
        raise ValueError("a class has fewer source images than images_per_name")

    steps = NameBalancedBatchPlan.steps_for_images(len(h5_index), plan)
    for _ in range(steps):
        selected_classes = rng.choice(
            classes, size=plan.names_per_batch, replace=False
        )
        sampled = []
        for label in selected_classes:
            sampled.extend(
                rng.choice(
                    by_class[label], size=plan.images_per_name, replace=False
                ).tolist()
            )
        yield np.asarray(sampled, dtype=np.int64)


def triplet_indices(batch_h5_index, labels, rng):
    """Sample one unrestricted random positive and negative per batch anchor."""
    batch_h5_index = np.asarray(batch_h5_index)
    labels = np.asarray(labels)
    if batch_h5_index.ndim != 1 or labels.ndim != 1:
        raise ValueError("triplet identities and labels must be one-dimensional")
    if len(batch_h5_index) != len(labels):
        raise ValueError("triplet identities and labels must have equal length")
    if len(np.unique(batch_h5_index)) != len(batch_h5_index):
        raise ValueError("a training batch cannot repeat source identities")

    anchors = np.arange(len(labels), dtype=np.int64)
    positives = np.empty(len(labels), dtype=np.int64)
    negatives = np.empty(len(labels), dtype=np.int64)
    for anchor in anchors:
        same_name = np.flatnonzero(labels == labels[anchor])
        same_name = same_name[same_name != anchor]
        other_name = np.flatnonzero(labels != labels[anchor])
        if len(same_name) == 0 or len(other_name) == 0:
            raise ValueError("every anchor requires one positive and one negative")
        positives[anchor] = rng.choice(same_name)
        negatives[anchor] = rng.choice(other_name)
    return anchors, positives, negatives


def supcon_positive_mask(labels):
    """Return the same-name positive mask, excluding anchor self-pairs."""
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    return (labels[:, None] == labels[None, :]) & ~np.eye(
        len(labels), dtype=bool
    )


def _loss_contract_builders(config):
    return {
        "triplet_margin": lambda: {
            "loss": "triplet_margin",
            "negative_sampling": (
                "uniform unrestricted within other-name batch members"
            ),
            "positive_sampling": "uniform other same-name batch member",
        },
        "arcface": lambda: {
            "loss": "arcface",
            "classifier_classes": config.num_classes,
            "temporary_head": True,
        },
        "supcon": lambda: {
            "loss": "supcon",
            "positives": "same-name batch members excluding the anchor",
            "implementation": "pytorch-metric-learning SupConLoss",
        },
    }


def loss_contract(candidate, config=DEFAULT_TRAINING_CONFIG):
    """Return the externally recorded fitting behavior for a candidate loss."""
    try:
        return _loss_contract_builders(config)[candidate.loss]()
    except KeyError as error:
        raise ValueError(f"unknown loss {candidate.loss!r}") from error


_LOSS_SELECTION_TIES = {
    "triplet_margin": lambda hyperparameters: (hyperparameters["margin"],),
    "arcface": lambda hyperparameters: (
        hyperparameters["margin"],
        hyperparameters["scale"],
    ),
    "supcon": lambda hyperparameters: (hyperparameters["temperature"],),
}


def _selection_key(row):
    hyperparameters = (
        row["hyperparameters"]
        if isinstance(row.get("hyperparameters"), dict)
        else json.loads(row["hyperparameters_json"])
    )
    try:
        tie = _LOSS_SELECTION_TIES[row["loss"]](hyperparameters)
    except KeyError as error:
        raise ValueError(f"unknown loss {row['loss']!r}") from error
    return (
        -float(row["best_validation_triplet_accuracy"]),
        *tie,
        row["trial_id"],
    )


def select_candidates(rows):
    """Mark one deterministically selected row per loss without retraining."""
    rows = [dict(row) for row in rows]
    for loss in LOSSES:
        members = [row for row in rows if row["loss"] == loss]
        if not members:
            raise ValueError(f"selection.{loss}: no candidates")
        winner = min(members, key=_selection_key)
        for row in members:
            row["selected"] = "true" if row is winner else "false"
    return rows


def trial_checkpoint_path(repository_root, trial):
    """Return the sole local-only best-checkpoint path for a candidate fit."""
    trial_id = trial["trial_id"] if isinstance(trial, dict) else str(trial)
    if not trial_id or Path(trial_id).name != trial_id:
        raise ValueError("trial_id must be a non-empty single path component")
    return (
        Path(repository_root)
        / "models"
        / "metric_learning"
        / trial_id
        / "best.pt"
    )


SELECTED_EMBEDDING_FILENAMES = {
    "triplet_margin": "triplet_margin_selected.npz",
    "arcface": "arcface_selected.npz",
    "supcon": "supcon_selected.npz",
}


def read_selected_trials(trial_results_path):
    """Read the complete ledger and return exactly the selected 3/4/3-grid rows."""
    from .handoff import validate_trial_ledger

    ledger = validate_trial_ledger(trial_results_path)
    selected_ids = ledger["selected_trials"]
    rows_by_id = {row["trial_id"]: row for row in ledger["rows"]}
    return {
        loss: rows_by_id[selected_ids[loss]]
        for loss in LOSSES
    }


def export_selected_embeddings(
    *,
    repository_root,
    manifest_path,
    hdf5_path,
    trial_results_path,
    package_dir,
    model_loader=None,
    embedding_exporter=None,
    batch_size=128,
    device="cpu",
    expected_dimension=EMBEDDING_DIMENSION,
):
    """Export only the three selected local checkpoints as stable NPZ files."""
    selected = read_selected_trials(trial_results_path)
    checkpoint_paths = {
        loss: trial_checkpoint_path(repository_root, trial)
        for loss, trial in selected.items()
    }
    for path in checkpoint_paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"selected checkpoint missing: {path}")

    if model_loader is None:
        from .iresnet import load_iresnet100

        model_loader = load_iresnet100
    if embedding_exporter is None:
        embedding_exporter = export_baseline_embeddings

    package_dir = Path(package_dir)
    outputs = {}
    for loss in LOSSES:
        model = model_loader(checkpoint_paths[loss], device)
        output_path = (
            package_dir
            / "embeddings"
            / SELECTED_EMBEDDING_FILENAMES[loss]
        )
        embedding_exporter(
            model,
            hdf5_path=hdf5_path,
            manifest_path=manifest_path,
            output_path=output_path,
            batch_size=batch_size,
            device=device,
            expected_dimension=expected_dimension,
        )
        outputs[output_path.name] = output_path
    return outputs


def write_trial_ledger(path, rows):
    """Write a deterministic complete ledger after validating all 10 candidates."""
    rows = [dict(row) for row in rows]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=TRIAL_LEDGER_FIELDS, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)
    from .handoff import validate_trial_ledger

    return validate_trial_ledger(path)


def initialize_trial_ledger(path):
    """Create an empty ledger before a new ten-fit search."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        csv.DictWriter(stream, fieldnames=TRIAL_LEDGER_FIELDS).writeheader()
    return path


def append_trial_row(path, row):
    """Persist a completed fit immediately so interrupted searches remain visible."""
    path = Path(path)
    with path.open("a", newline="", encoding="utf-8") as stream:
        csv.DictWriter(
            stream, fieldnames=TRIAL_LEDGER_FIELDS, extrasaction="ignore"
        ).writerow(row)
    return path

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
