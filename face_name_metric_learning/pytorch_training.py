"""PyTorch fitting implementation consumed by the candidate-training notebook."""

import gc
import json
from pathlib import Path

import h5py
import numpy as np

from .iresnet import load_iresnet100
from .training import (
    DEFAULT_TRAINING_CONFIG,
    BestEpochSelector,
    NameBalancedBatchPlan,
    append_trial_row,
    candidate_configurations,
    child_seed,
    initialize_trial_ledger,
    select_candidates,
    shuffled_name_balanced_batches,
    trial_checkpoint_path,
    triplet_indices,
    write_trial_ledger,
)
from face_name_metrics import blocked_triplet_accuracy


def _require_cuda(torch):
    if not torch.cuda.is_available():
        raise RuntimeError("candidate training requires a CUDA device for AMP")
    return torch.device("cuda")


def _seed_runtime(torch, seed):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _load_manifest_arrays(manifest_rows, split):
    selected = [row for row in manifest_rows if row["split"] == split]
    return (
        np.asarray([row["h5_index"] for row in selected], dtype=np.int64),
        np.asarray([row["label"] for row in selected], dtype=np.int64),
    )


def _loss_module(candidate, config):
    from pytorch_metric_learning import losses

    builders = {
        "triplet_margin": lambda: None,
        "arcface": lambda: losses.ArcFaceLoss(
            num_classes=config.num_classes,
            embedding_size=config.embedding_dimension,
            margin=candidate.hyperparameters["margin"],
            scale=candidate.hyperparameters["scale"],
        ),
        "supcon": lambda: losses.SupConLoss(
            temperature=candidate.hyperparameters["temperature"]
        ),
    }
    try:
        return builders[candidate.loss]()
    except KeyError as error:
        raise ValueError(f"unknown loss {candidate.loss!r}") from error


def _training_loss(torch, candidate, criterion, embeddings, labels, loss_rng):
    import torch.nn.functional as functional

    def criterion_loss():
        return criterion(embeddings, labels)

    def triplet_loss():
        anchors, positives, negatives = triplet_indices(
            np.arange(len(labels), dtype=np.int64),
            labels.detach().cpu().numpy(),
            loss_rng,
        )
        normalized = functional.normalize(embeddings, dim=1)
        positive = (normalized[anchors] * normalized[positives]).sum(dim=1)
        negative = (normalized[anchors] * normalized[negatives]).sum(dim=1)
        return functional.relu(
            negative - positive + candidate.hyperparameters["margin"]
        ).mean()

    calculators = {
        "triplet_margin": triplet_loss,
        "arcface": criterion_loss,
        "supcon": criterion_loss,
    }
    try:
        return calculators[candidate.loss]()
    except KeyError as error:
        raise ValueError(f"unknown loss {candidate.loss!r}") from error


def _read_training_batch(hdf5, source_positions, augmentation_rng, flip_probability):
    sorted_positions = np.sort(source_positions)
    batch = np.asarray(hdf5["images"][sorted_positions.tolist()])
    batch = batch[np.searchsorted(sorted_positions, source_positions)]
    flips = augmentation_rng.random(len(batch)) < flip_probability
    batch[flips] = batch[flips, :, :, ::-1]
    return batch


def _validation_embeddings(torch, model, hdf5_path, validation_indices, batch_size):
    model.eval()
    batches = []
    with h5py.File(hdf5_path, "r") as hdf5, torch.inference_mode():
        for start in range(0, len(validation_indices), batch_size):
            positions = validation_indices[start : start + batch_size]
            sorted_positions = np.sort(positions)
            images = np.asarray(hdf5["images"][sorted_positions.tolist()])
            images = images[np.searchsorted(sorted_positions, positions)]
            images = torch.from_numpy(images).to("cuda")
            embeddings = model(images)
            batches.append(
                torch.nn.functional.normalize(embeddings, dim=1)
                .float()
                .cpu()
                .numpy()
            )
    return np.concatenate(batches, axis=0)


def _probe_plan(
    torch,
    checkpoint_path,
    candidate,
    config,
    training_hdf5,
    train_h5_index,
    train_labels,
):
    plans = (NameBalancedBatchPlan.names_16_by_8(), NameBalancedBatchPlan.names_8_by_8())
    errors = []
    for plan in plans:
        model = None
        criterion = None
        try:
            model = load_iresnet100(checkpoint_path, "cuda")
            criterion = _loss_module(candidate, config)
            if criterion is not None:
                criterion.to("cuda")
            batch_positions = next(
                shuffled_name_balanced_batches(
                    train_h5_index,
                    train_labels,
                    plan,
                    np.random.default_rng(child_seed(candidate, "memory-probe")),
                )
            )
            source_positions = train_h5_index[batch_positions]
            with torch.amp.autocast("cuda"):
                sorted_positions = np.sort(source_positions)
                images = np.asarray(
                    training_hdf5["images"][sorted_positions.tolist()]
                )
                images = images[np.searchsorted(sorted_positions, source_positions)]
                images = torch.from_numpy(images).to("cuda")
                labels = torch.from_numpy(train_labels[batch_positions]).to("cuda")
                embeddings = model(images)
                loss = _training_loss(
                    torch,
                    candidate,
                    criterion,
                    embeddings,
                    labels,
                    np.random.default_rng(len(labels)),
                )
            loss.backward()
            model.zero_grad(set_to_none=True)
            if criterion is not None:
                criterion.zero_grad(set_to_none=True)
            return plan
        except torch.cuda.OutOfMemoryError as error:
            errors.append(error)
            torch.cuda.empty_cache()
        finally:
            del model, criterion
            gc.collect()
    raise RuntimeError("both 16x8 and 8x8 fitting batches exceed CUDA memory") from errors[-1]


def train_candidate(
    candidate,
    *,
    repository_root,
    manifest_path,
    hdf5_path,
    pretrained_checkpoint_path,
    manifest_rows,
    config=DEFAULT_TRAINING_CONFIG,
):
    """Fit one candidate once and retain only its best-epoch backbone checkpoint."""
    try:
        import torch
    except ImportError as error:
        raise ImportError("candidate training requires PyTorch") from error
    from torch import nn
    from pytorch_metric_learning import __version__ as metric_learning_version

    device = _require_cuda(torch)
    _seed_runtime(torch, child_seed(candidate, "runtime-initialization"))
    repository_root = Path(repository_root)
    hdf5_path = Path(hdf5_path)
    checkpoint_path = trial_checkpoint_path(repository_root, candidate)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    train_h5_index, train_labels = _load_manifest_arrays(manifest_rows, "train")
    validation_h5_index, _ = _load_manifest_arrays(manifest_rows, "validation")
    validation_names = np.asarray(
        [
            row["first_name"]
            for row in manifest_rows
            if row["split"] == "validation"
        ]
    )

    with h5py.File(hdf5_path, "r") as hdf5:
        if train_h5_index.size == 0:
            raise ValueError("training manifest is empty")
        plan = _probe_plan(
            torch,
            pretrained_checkpoint_path,
            candidate,
            config,
            hdf5,
            train_h5_index,
            train_labels,
        )

    model = load_iresnet100(pretrained_checkpoint_path, device)
    criterion = _loss_module(candidate, config)
    parameters = list(model.parameters())
    if criterion is not None:
        criterion.to(device)
        parameters.extend(criterion.parameters())
    optimizer = torch.optim.Adam(
        parameters,
        lr=config.learning_rate,
        betas=config.betas,
        weight_decay=config.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    steps_per_epoch = NameBalancedBatchPlan.steps_for_images(
        len(train_h5_index), plan
    )
    selection = BestEpochSelector(
        patience=config.early_stop_patience,
        minimum_improvement=config.minimum_improvement,
    )
    early_stopped = False
    images_seen = 0

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        batch_rng = np.random.default_rng(
            child_seed(candidate, f"batch-sampling/epoch-{epoch}")
        )
        augmentation_rng = np.random.default_rng(
            child_seed(candidate, f"augmentation/epoch-{epoch}")
        )
        loss_rng = np.random.default_rng(
            child_seed(candidate, f"loss-sampling/epoch-{epoch}")
        )
        with h5py.File(hdf5_path, "r") as hdf5:
            for batch_positions in shuffled_name_balanced_batches(
                train_h5_index, train_labels, plan, batch_rng
            ):
                source_positions = train_h5_index[batch_positions]
                images = _read_training_batch(
                    hdf5,
                    source_positions,
                    augmentation_rng,
                    config.flip_probability,
                )
                labels = torch.from_numpy(train_labels[batch_positions]).to(device)
                images = torch.from_numpy(images).to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda"):
                    embeddings = model(images)
                    loss = _training_loss(
                        torch, candidate, criterion, embeddings, labels, loss_rng
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                images_seen += len(images)

        validation_embeddings = _validation_embeddings(
            torch, model, hdf5_path, validation_h5_index, 256
        )
        accuracy = blocked_triplet_accuracy(
            validation_embeddings,
            validation_h5_index,
            validation_names,
            block_size=config.validation_block_size,
        )
        improved = selection.observe(epoch, accuracy)
        if improved:
            torch.save(
                {
                    "trial_id": candidate.trial_id,
                    "loss": candidate.loss,
                    "hyperparameters": dict(candidate.hyperparameters),
                    "seed": candidate.seed,
                    "best_epoch": epoch,
                    "best_validation_triplet_accuracy": float(accuracy),
                    "backbone_state_dict": model.state_dict(),
                },
                checkpoint_path,
            )
        if selection.should_stop:
            early_stopped = True
            break

    if not checkpoint_path.exists():
        raise RuntimeError(f"candidate {candidate.trial_id} produced no checkpoint")
    notes = [
        f"batch={plan}",
        "augmentation=horizontal_flip_p0.5",
        f"pytorch_metric_learning={metric_learning_version}",
    ]
    notes.append(
        {
            "triplet_margin": "sampling=unrestricted_random_negative",
            "arcface": "head=100_class_temporary",
            "supcon": "supcon=same_name_batch_positives",
        }[candidate.loss]
    )
    return {
        "trial_id": candidate.trial_id,
        "loss": candidate.loss,
        "hyperparameters_json": json.dumps(
            candidate.hyperparameters, sort_keys=True, separators=(",", ":")
        ),
        "seed": candidate.seed,
        "epochs_attempted": epoch,
        "best_epoch": selection.best_epoch,
        "best_validation_triplet_accuracy": selection.best_accuracy,
        "early_stopped": str(early_stopped).lower(),
        "images_seen": images_seen,
        "steps_per_epoch": steps_per_epoch,
        "device": str(device),
        "selected": "false",
        "notes": "; ".join(notes),
    }


def train_and_select_candidates(
    *,
    repository_root,
    manifest_path,
    hdf5_path,
    trial_results_path,
    pretrained_checkpoint_path,
    manifest_rows,
    config=DEFAULT_TRAINING_CONFIG,
):
    """Run each candidate exactly once, persist the ledger, and mark selections."""
    candidates = candidate_configurations(config.root_seed)
    if len(candidates) != 10:
        raise RuntimeError("candidate grid must contain exactly ten configurations")
    initialize_trial_ledger(trial_results_path)
    rows = []
    for candidate in candidates:
        row = train_candidate(
            candidate,
            repository_root=repository_root,
            manifest_path=manifest_path,
            hdf5_path=hdf5_path,
            pretrained_checkpoint_path=pretrained_checkpoint_path,
            manifest_rows=manifest_rows,
            config=config,
        )
        append_trial_row(trial_results_path, row)
        rows.append(row)
    selected = select_candidates(rows)
    write_trial_ledger(trial_results_path, selected)
    return selected
