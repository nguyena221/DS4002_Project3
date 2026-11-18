"""Reusable configuration, sampling, selection, and ledger behavior for training."""

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


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
        if self.loss == "triplet_margin":
            suffix = f"margin{_id_number(self.hyperparameters['margin'])}"
        elif self.loss == "arcface":
            suffix = (
                f"m{_id_number(self.hyperparameters['margin'])}"
                f"_s{self.hyperparameters['scale']}"
            )
        elif self.loss == "supcon":
            suffix = f"t{_id_number(self.hyperparameters['temperature'])}"
        else:
            raise ValueError(f"unknown loss {self.loss!r}")
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


def child_seed(candidate, concern):
    """Derive a stable 32-bit seed for one independently named RNG concern."""
    payload = json.dumps(
        {
            "root_seed": int(candidate.seed),
            "trial_id": candidate.trial_id,
            "concern": str(concern),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


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


def loss_contract(candidate, config=DEFAULT_TRAINING_CONFIG):
    """Return the externally recorded fitting behavior for a candidate loss."""
    if candidate.loss == "triplet_margin":
        return {
            "loss": candidate.loss,
            "negative_sampling": "uniform unrestricted within other-name batch members",
            "positive_sampling": "uniform other same-name batch member",
        }
    if candidate.loss == "arcface":
        return {
            "loss": candidate.loss,
            "classifier_classes": config.num_classes,
            "temporary_head": True,
        }
    if candidate.loss == "supcon":
        return {
            "loss": candidate.loss,
            "positives": "same-name batch members excluding the anchor",
            "implementation": "pytorch-metric-learning SupConLoss",
        }
    raise ValueError(f"unknown loss {candidate.loss!r}")


def _selection_key(row):
    hyperparameters = (
        row["hyperparameters"]
        if isinstance(row.get("hyperparameters"), dict)
        else json.loads(row["hyperparameters_json"])
    )
    loss = row["loss"]
    if loss == "triplet_margin":
        tie = (hyperparameters["margin"],)
    elif loss == "arcface":
        tie = (hyperparameters["margin"], hyperparameters["scale"])
    elif loss == "supcon":
        tie = (hyperparameters["temperature"],)
    else:
        raise ValueError(f"unknown loss {loss!r}")
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
