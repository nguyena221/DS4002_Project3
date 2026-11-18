"""Validation and I/O for face-name metric-learning handoff artifacts."""

import csv
import hashlib
import json
from datetime import datetime, timezone
from collections import Counter
from pathlib import Path

import h5py
import numpy as np


EXPECTED_SPLIT_COUNTS = {"train": 45_078, "validation": 9_660, "test": 9_660}
EXPECTED_CLASS_COUNT = 100
EMBEDDING_DIMENSION = 512
NORM_TOLERANCE = 1e-5
EMBEDDING_KEYS = {
    "validation_h5_index",
    "validation_embeddings",
    "test_h5_index",
    "test_embeddings",
}


def _require(condition: bool, field: str, detail: str) -> None:
    if not condition:
        raise ValueError(f"{field}: {detail}")


def _decode(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def read_manifest(path):
    """Read the committed face manifest with validated scalar identities."""
    path = Path(path)
    required = {"h5_index", "filename", "first_name", "label", "split"}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        _require(reader.fieldnames is not None, "manifest.header", "missing header")
        missing = required.difference(reader.fieldnames or ())
        _require(not missing, "manifest.header", f"missing required fields {sorted(missing)}")
        rows = list(reader)

    for row_number, row in enumerate(rows, start=2):
        try:
            row["h5_index"] = int(row["h5_index"])
            row["label"] = int(row["label"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"manifest.row[{row_number}]: invalid integer identity") from error
    return rows


def validate_manifest_hdf5(
    manifest_path,
    hdf5_path,
    *,
    expected_split_counts=EXPECTED_SPLIT_COUNTS,
    expected_class_count=EXPECTED_CLASS_COUNT,
):
    """Validate that the CSV and HDF5 describe the same prepared dataset."""
    rows = read_manifest(manifest_path)
    indices = [row["h5_index"] for row in rows]
    _require(indices, "manifest.h5_index", "manifest is empty")
    _require(len(indices) == len(set(indices)), "manifest.h5_index", "contains duplicates")
    expected_indices = np.arange(len(rows), dtype=np.int64)
    _require(
        np.array_equal(np.sort(np.asarray(indices, dtype=np.int64)), expected_indices),
        "manifest.h5_index",
        "must uniquely cover HDF5 row indices 0 through N-1",
    )

    split_counts = dict(Counter(row["split"] for row in rows))
    expected_split_counts = dict(expected_split_counts)
    _require(
        split_counts == expected_split_counts,
        "manifest.split",
        f"counts are {split_counts}, expected {expected_split_counts}",
    )
    class_count = len({row["first_name"] for row in rows})
    _require(
        class_count == expected_class_count,
        "manifest.first_name",
        f"class count is {class_count}, expected {expected_class_count}",
    )

    with h5py.File(hdf5_path, "r") as h5:
        _require(bool(h5.attrs.get("complete", False)), "faces.h5.complete", "must be true")
        _require(
            all(key in h5 for key in ("filenames", "labels", "split", "class_names", "split_indices")),
            "faces.h5",
            "missing required datasets",
        )
        filenames = [_decode(value) for value in h5["filenames"][:]]
        labels = np.asarray(h5["labels"][:])
        splits = np.asarray(h5["split"][:])
        class_names = [_decode(value) for value in h5["class_names"][:]]
        _require(
            len(rows) == len(filenames) == len(labels) == len(splits),
            "faces.h5.images",
            "row count does not match manifest",
        )
        for row in rows:
            index = row["h5_index"]
            _require(filenames[index] == row["filename"], f"manifest.row[{index}].filename", "does not match HDF5")
            _require(int(labels[index]) == row["label"], f"manifest.row[{index}].label", "does not match HDF5")
            label = row["label"]
            _require(
                0 <= label < len(class_names) and class_names[label] == row["first_name"],
                f"manifest.row[{index}].first_name",
                "does not match HDF5 class_names at label",
            )
            split_code = int(splits[index])
            expected_code = {"train": 0, "validation": 1, "test": 2}.get(row["split"])
            _require(expected_code is not None, f"manifest.row[{index}].split", f"unknown split {row['split']!r}")
            _require(split_code == expected_code, f"manifest.row[{index}].split", "does not match HDF5 split")

        for split, expected_indices_for_split in {
            split: sorted(row["h5_index"] for row in rows if row["split"] == split)
            for split in expected_split_counts
        }.items():
            _require(
                split in h5["split_indices"],
                f"split_indices.{split}",
                "missing dataset",
            )
            actual = np.asarray(h5[f"split_indices/{split}"][:], dtype=np.int64)
            _require(
                np.array_equal(np.sort(actual), np.asarray(expected_indices_for_split, dtype=np.int64)),
                f"split_indices.{split}",
                "does not exactly cover manifest split",
            )

    return {
        "rows": len(rows),
        "split_counts": split_counts,
        "class_count": class_count,
    }


def _validate_embedding_split(
    split,
    h5_index,
    embeddings,
    manifest_rows,
    *,
    expected_dimension,
    norm_tolerance,
):
    h5_index = np.asarray(h5_index)
    embeddings = np.asarray(embeddings)
    _require(h5_index.ndim == 1, f"{split}_h5_index.shape", f"has shape {h5_index.shape}, expected 1-D")
    _require(h5_index.dtype == np.dtype(np.int64), f"{split}_h5_index.dtype", f"is {h5_index.dtype}, expected int64")
    _require(
        embeddings.ndim == 2 and embeddings.shape[1] == expected_dimension,
        f"{split}_embeddings.shape",
        f"has shape {embeddings.shape}, expected (N, {expected_dimension})",
    )
    _require(embeddings.shape[0] == h5_index.shape[0], f"{split}_embeddings.shape", "row count does not match index count")
    _require(
        embeddings.dtype == np.dtype(np.float32),
        f"{split}_embeddings.dtype",
        f"is {embeddings.dtype}, expected float32",
    )
    _require(len(np.unique(h5_index)) == len(h5_index), f"{split}_h5_index.unique", "contains duplicate indices")
    expected = sorted(row["h5_index"] for row in manifest_rows if row["split"] == split)
    _require(
        np.array_equal(np.sort(h5_index), np.asarray(expected, dtype=np.int64)),
        f"{split}_h5_index.coverage",
        f"must exactly cover manifest {split} split",
    )
    _require(np.isfinite(embeddings).all(), f"{split}_embeddings.finite", "contains non-finite values")
    norms = np.linalg.norm(embeddings, axis=1)
    _require(
        np.all(np.abs(norms - 1.0) <= norm_tolerance),
        f"{split}_embeddings.norm",
        f"L2 norms must be within {norm_tolerance} of 1",
    )


def validate_embedding_package(
    package,
    *,
    manifest_path,
    expected_dimension=EMBEDDING_DIMENSION,
    norm_tolerance=NORM_TOLERANCE,
):
    """Validate the four-array embedding NPZ contract."""
    rows = read_manifest(manifest_path)
    _require(set(package) == EMBEDDING_KEYS, "embedding_package.keys", f"expected exactly {sorted(EMBEDDING_KEYS)}")
    for split in ("validation", "test"):
        _validate_embedding_split(
            split,
            package[f"{split}_h5_index"],
            package[f"{split}_embeddings"],
            rows,
            expected_dimension=expected_dimension,
            norm_tolerance=norm_tolerance,
        )
    return package


def write_embedding_package(
    path,
    validation_h5_index,
    validation_embeddings,
    test_h5_index,
    test_embeddings,
    *,
    manifest_path,
    expected_dimension=EMBEDDING_DIMENSION,
    norm_tolerance=NORM_TOLERANCE,
):
    """Validate and write a compressed embedding handoff package."""
    package = {
        "validation_h5_index": np.asarray(validation_h5_index),
        "validation_embeddings": np.asarray(validation_embeddings),
        "test_h5_index": np.asarray(test_h5_index),
        "test_embeddings": np.asarray(test_embeddings),
    }
    validate_embedding_package(
        package,
        manifest_path=manifest_path,
        expected_dimension=expected_dimension,
        norm_tolerance=norm_tolerance,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **package)
    return path


def load_embedding_package(
    path,
    *,
    manifest_path,
    expected_dimension=EMBEDDING_DIMENSION,
    norm_tolerance=NORM_TOLERANCE,
):
    """Load and validate an embedding package without retaining an open NPZ."""
    with np.load(path, allow_pickle=False) as archive:
        package = {key: archive[key].copy() for key in archive.files}
    return validate_embedding_package(
        package,
        manifest_path=manifest_path,
        expected_dimension=expected_dimension,
        norm_tolerance=norm_tolerance,
    )


def align_embeddings(h5_index, embeddings, target_h5_index):
    """Return embeddings in target index order, never relying on row position."""
    source = np.asarray(h5_index)
    values = np.asarray(embeddings)
    target = np.asarray(target_h5_index)
    _require(source.ndim == 1, "h5_index.shape", "source indices must be 1-D")
    _require(target.ndim == 1, "target_h5_index.shape", "target indices must be 1-D")
    _require(len(np.unique(source)) == len(source), "h5_index.unique", "source contains duplicates")
    _require(len(np.unique(target)) == len(target), "target_h5_index.unique", "target contains duplicates")
    _require(values.ndim == 2 and values.shape[0] == source.shape[0], "embeddings.shape", "rows must match indices")
    _require(
        np.array_equal(np.sort(source), np.sort(target)),
        "h5_index.coverage",
        "source and target identities differ",
    )
    positions = {int(index): position for position, index in enumerate(source)}
    return values[[positions[int(index)] for index in target]]


TRIAL_FIELDS = {
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
}
EXPECTED_CANDIDATE_GRID = {
    "triplet_margin": {("margin", 0.1), ("margin", 0.2), ("margin", 0.4)},
    "arcface": {
        ("margin", 0.2, "scale", 30),
        ("margin", 0.2, "scale", 64),
        ("margin", 0.4, "scale", 30),
        ("margin", 0.4, "scale", 64),
    },
    "supcon": {("temperature", 0.07), ("temperature", 0.1), ("temperature", 0.2)},
}


def _canonical_loss(value):
    compact = "".join(character for character in str(value).lower() if character.isalnum())
    aliases = {
        "tripletmargin": "triplet_margin",
        "tripletmarginloss": "triplet_margin",
        "arcface": "arcface",
        "supcon": "supcon",
        "supervisedcontrastive": "supcon",
    }
    return aliases.get(compact)


def _csv_boolean(value, field):
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"{field}: expected true or false, got {value!r}")


def validate_trial_ledger(path):
    """Validate the complete 10-candidate trial ledger and its selections."""
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = TRIAL_FIELDS.difference(reader.fieldnames or ())
        _require(not missing, "trial_results.header", f"missing fields {sorted(missing)}")
        rows = list(reader)
    _require(len(rows) == 10, "trial_results.count", f"is {len(rows)}, expected 10")

    seen_trial_ids = set()
    configurations = {loss: [] for loss in EXPECTED_CANDIDATE_GRID}
    selected_trials = {loss: [] for loss in EXPECTED_CANDIDATE_GRID}
    parsed_rows = []
    for row_number, raw in enumerate(rows, start=2):
        trial_id = (raw["trial_id"] or "").strip()
        _require(trial_id, f"trial_results.row[{row_number}].trial_id", "must not be empty")
        _require(trial_id not in seen_trial_ids, "trial_results.trial_id", f"duplicate {trial_id!r}")
        seen_trial_ids.add(trial_id)

        loss = _canonical_loss(raw["loss"])
        _require(loss is not None, f"trial_results.row[{row_number}].loss", f"unknown loss {raw['loss']!r}")
        try:
            hyperparameters = json.loads(raw["hyperparameters_json"])
        except json.JSONDecodeError as error:
            raise ValueError(f"trial_results.row[{row_number}].hyperparameters_json: invalid JSON") from error
        _require(
            isinstance(hyperparameters, dict),
            f"trial_results.row[{row_number}].hyperparameters_json",
            "must be a JSON object",
        )
        expected_keys = {
            "triplet_margin": {"margin"},
            "arcface": {"margin", "scale"},
            "supcon": {"temperature"},
        }[loss]
        _require(
            set(hyperparameters) == expected_keys,
            f"trial_results.row[{row_number}].hyperparameters_json",
            f"expected keys {sorted(expected_keys)}",
        )
        _require(
            all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and np.isfinite(value)
                for value in hyperparameters.values()
            ),
            f"trial_results.row[{row_number}].hyperparameters_json",
            "values must be finite numbers",
        )
        configuration = tuple(
            item
            for key, value in sorted(hyperparameters.items())
            for item in (key, value)
        )
        configurations[loss].append(configuration)

        selected = _csv_boolean(raw["selected"], f"trial_results.row[{row_number}].selected")
        early_stopped = _csv_boolean(raw["early_stopped"], f"trial_results.row[{row_number}].early_stopped")
        if selected:
            selected_trials[loss].append(trial_id)
        for integer_field in ("seed", "epochs_attempted", "best_epoch", "images_seen", "steps_per_epoch"):
            try:
                value = int(raw[integer_field])
            except (TypeError, ValueError) as error:
                raise ValueError(f"trial_results.row[{row_number}].{integer_field}: expected integer") from error
            _require(value >= 0, f"trial_results.row[{row_number}].{integer_field}", "must not be negative")
        try:
            accuracy = float(raw["best_validation_triplet_accuracy"])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"trial_results.row[{row_number}].best_validation_triplet_accuracy: expected number"
            ) from error
        _require(
            np.isfinite(accuracy) and 0 <= accuracy <= 1,
            f"trial_results.row[{row_number}].best_validation_triplet_accuracy",
            "must be finite and between 0 and 1",
        )
        parsed = dict(raw)
        parsed.update(loss=loss, hyperparameters=hyperparameters, selected=selected, early_stopped=early_stopped)
        parsed_rows.append(parsed)

    for loss, expected in EXPECTED_CANDIDATE_GRID.items():
        actual = configurations[loss]
        _require(len(actual) == len(expected), f"trial_results.{loss}.count", f"is {len(actual)}, expected {len(expected)}")
        _require(set(actual) == expected, f"trial_results.{loss}.grid", f"is {actual}, expected {sorted(expected)}")
        _require(
            len(selected_trials[loss]) == 1,
            f"trial_results.{loss}.selected",
            f"has {len(selected_trials[loss])} selected rows, expected exactly one",
        )

    return {
        "rows": parsed_rows,
        "selected_trials": {loss: trial_ids[0] for loss, trial_ids in selected_trials.items()},
    }


MODEL_FILENAMES = {
    "baseline_iresnet100": "baseline_iresnet100.npz",
    "triplet_margin_selected": "triplet_margin_selected.npz",
    "arcface_selected": "arcface_selected.npz",
    "supcon_selected": "supcon_selected.npz",
}
VERSION_FIELDS = {
    "python",
    "pytorch",
    "cuda",
    "pytorch_metric_learning",
    "pytorch_insightface",
}
TRIPLET_ACCURACY_DEFINITION = (
    "correct_predictions / total_valid_comparisons, where cosine similarity is "
    "strictly greater for the positive than the negative and ties are incorrect"
)
SIMILARITY_LIFT_DEFINITION = (
    "fine_tuned_triplet_accuracy - baseline_triplet_accuracy, expressed as a "
    "signed difference in percentage points"
)


def sha256_file(path):
    """Return the lowercase SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _validate_utc_timestamp(value, field):
    _require(isinstance(value, str) and value.endswith("Z"), field, "must be an ISO UTC timestamp ending in Z")
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError(f"{field}: invalid ISO UTC timestamp") from error


def _manifest_inputs(
    package_dir,
    manifest_path,
    hdf5_path,
    trial_results_path,
):
    package_dir = Path(package_dir)
    return {
        "faces.h5": Path(hdf5_path),
        "manifest.csv": Path(manifest_path),
        "trial_results.csv": Path(trial_results_path),
        **{
            f"embeddings/{filename}": package_dir / "embeddings" / filename
            for filename in MODEL_FILENAMES.values()
        },
    }


def build_handoff_manifest(
    package_dir,
    *,
    manifest_path,
    hdf5_path,
    trial_results_path,
    versions,
    pretrained_weights,
    device,
    child_seed_policy,
    split_counts=EXPECTED_SPLIT_COUNTS,
    class_count=EXPECTED_CLASS_COUNT,
    root_seed=4002,
    expected_dimension=EMBEDDING_DIMENSION,
    norm_tolerance=NORM_TOLERANCE,
    generated_at=None,
    git_commit=None,
    audit=None,
):
    """Build schema-version-1 metadata after validating all handoff inputs."""
    package_dir = Path(package_dir)
    rows = read_manifest(manifest_path)
    observed_split_counts = Counter(row["split"] for row in rows)
    actual_split_counts = {split: observed_split_counts[split] for split in ("train", "validation", "test")}
    actual_class_count = len({row["first_name"] for row in rows})
    _require(actual_split_counts == dict(split_counts), "split_counts", f"are {actual_split_counts}, expected {split_counts}")
    _require(actual_class_count == class_count, "class_count", f"is {actual_class_count}, expected {class_count}")
    ledger = validate_trial_ledger(trial_results_path)
    inputs = _manifest_inputs(package_dir, manifest_path, hdf5_path, trial_results_path)
    for filename in MODEL_FILENAMES.values():
        load_embedding_package(
            package_dir / "embeddings" / filename,
            manifest_path=manifest_path,
            expected_dimension=expected_dimension,
            norm_tolerance=norm_tolerance,
        )

    generated_at = generated_at or _utc_now()
    audit = audit or {"status": "pass", "timestamp": _utc_now()}
    selected = ledger["selected_trials"]
    metadata = {
        "schema_version": 1,
        "generated_at": generated_at,
        "git_commit": git_commit,
        "checksums": {name: sha256_file(path) for name, path in inputs.items()},
        "versions": dict(versions),
        "pretrained_weights": dict(pretrained_weights),
        "root_seed": int(root_seed),
        "child_seed_policy": child_seed_policy,
        "split_counts": dict(split_counts),
        "class_count": int(class_count),
        "device": device,
        "selected_trials": selected,
        "models": {
            model: {
                "filename": f"embeddings/{filename}",
                "selected_trial_id": (
                    None
                    if model == "baseline_iresnet100"
                    else selected[model.removesuffix("_selected")]
                ),
            }
            for model, filename in MODEL_FILENAMES.items()
        },
        "metric_definitions": {
            "triplet_accuracy": TRIPLET_ACCURACY_DEFINITION,
            "similarity_lift": SIMILARITY_LIFT_DEFINITION,
        },
        "embedding_validation": {
            "embedding_dimension": int(expected_dimension),
            "embedding_dtype": "float32",
            "index_dtype": "int64",
            "finite": True,
            "unit_norm_tolerance": float(norm_tolerance),
            "index_coverage": "exactly once per manifest split",
        },
        "audit": dict(audit),
    }
    return validate_handoff_manifest(
        metadata,
        package_dir,
        manifest_path=manifest_path,
        hdf5_path=hdf5_path,
        trial_results_path=trial_results_path,
        expected_dimension=expected_dimension,
        norm_tolerance=norm_tolerance,
    )


def build_handoff_manifest_from_environment(
    package_dir,
    *,
    environment_record_path,
    manifest_path,
    hdf5_path,
    trial_results_path,
    child_seed_policy,
    split_counts=EXPECTED_SPLIT_COUNTS,
    class_count=EXPECTED_CLASS_COUNT,
    root_seed=4002,
    expected_dimension=EMBEDDING_DIMENSION,
    norm_tolerance=NORM_TOLERANCE,
    generated_at=None,
    git_commit=None,
    audit=None,
):
    """Build manifest metadata from the persisted setup and training records."""
    with Path(environment_record_path).open(encoding="utf-8") as stream:
        environment = json.load(stream)
    _require(isinstance(environment, dict), "environment_record", "must be an object")
    packages = environment.get("packages")
    _require(isinstance(packages, dict), "environment_record.packages", "must be an object")
    version_values = {
        "python": environment.get("python"),
        "pytorch": packages.get("pytorch"),
        "cuda": environment.get("cuda"),
        "pytorch_metric_learning": packages.get("pytorch_metric_learning"),
        "pytorch_insightface": packages.get("pytorch_insightface"),
    }
    versions = {
        name: str(value) if value is not None else ""
        for name, value in version_values.items()
    }
    weights = environment.get("pretrained_weights")
    _require(isinstance(weights, dict), "environment_record.pretrained_weights", "must be an object")

    ledger = validate_trial_ledger(trial_results_path)
    devices = {row["device"] for row in ledger["rows"]}
    _require(len(devices) == 1, "trial_results.device", f"expected one training device, got {sorted(devices)}")

    return build_handoff_manifest(
        package_dir,
        manifest_path=manifest_path,
        hdf5_path=hdf5_path,
        trial_results_path=trial_results_path,
        versions=versions,
        pretrained_weights=weights,
        device=devices.pop(),
        child_seed_policy=child_seed_policy,
        split_counts=split_counts,
        class_count=class_count,
        root_seed=root_seed,
        expected_dimension=expected_dimension,
        norm_tolerance=norm_tolerance,
        generated_at=generated_at,
        git_commit=git_commit,
        audit=audit,
    )


def validate_handoff_manifest(
    metadata,
    package_dir,
    *,
    manifest_path,
    hdf5_path,
    trial_results_path,
    expected_dimension=EMBEDDING_DIMENSION,
    norm_tolerance=NORM_TOLERANCE,
):
    """Validate metadata, source fingerprints, and all four model packages."""
    _require(isinstance(metadata, dict), "handoff_manifest", "must be an object")
    _require(metadata.get("schema_version") == 1, "schema_version", "expected 1")
    _validate_utc_timestamp(metadata.get("generated_at"), "generated_at")

    versions = metadata.get("versions")
    _require(isinstance(versions, dict), "versions", "must be an object")
    missing_versions = VERSION_FIELDS.difference(versions)
    _require(not missing_versions, "versions", f"missing fields {sorted(missing_versions)}")
    for field in VERSION_FIELDS:
        _require(isinstance(versions[field], str) and versions[field], f"versions.{field}", "must be a non-empty string")
    weights = metadata.get("pretrained_weights")
    _require(isinstance(weights, dict), "pretrained_weights", "must be an object")
    for field in ("identifier", "sha256"):
        _require(isinstance(weights.get(field), str) and weights[field], f"pretrained_weights.{field}", "must be a non-empty string")
    _require(isinstance(metadata.get("device"), str) and metadata["device"], "device", "must be a non-empty string")
    _require(isinstance(metadata.get("child_seed_policy"), str) and metadata["child_seed_policy"], "child_seed_policy", "must be a non-empty string")
    _require(metadata.get("root_seed") == 4002, "root_seed", "expected 4002")

    rows = read_manifest(manifest_path)
    observed_split_counts = Counter(row["split"] for row in rows)
    actual_split_counts = {split: observed_split_counts[split] for split in ("train", "validation", "test")}
    _require(metadata.get("split_counts") == actual_split_counts, "split_counts", f"does not match manifest counts {actual_split_counts}")
    actual_class_count = len({row["first_name"] for row in rows})
    _require(metadata.get("class_count") == actual_class_count, "class_count", f"is {metadata.get('class_count')}, expected {actual_class_count}")

    ledger = validate_trial_ledger(trial_results_path)
    _require(
        set(metadata.get("selected_trials", {})) == set(ledger["selected_trials"]),
        "selected_trials",
        f"expected exactly {sorted(ledger['selected_trials'])}",
    )
    for loss, expected_trial in ledger["selected_trials"].items():
        actual_trial = metadata.get("selected_trials", {}).get(loss)
        _require(actual_trial == expected_trial, f"selected_trials.{loss}", f"is {actual_trial!r}, expected {expected_trial!r}")

    inputs = _manifest_inputs(Path(package_dir), manifest_path, hdf5_path, trial_results_path)
    checksums = metadata.get("checksums")
    _require(isinstance(checksums, dict), "checksums", "must be an object")
    _require(set(checksums) == set(inputs), "checksums", f"expected exactly {sorted(inputs)}")
    for name, path in inputs.items():
        actual = sha256_file(path)
        _require(checksums[name] == actual, f"checksums.{name}", f"checksum mismatch: expected {actual}")

    models = metadata.get("models")
    _require(isinstance(models, dict) and set(models) == set(MODEL_FILENAMES), "models", "must describe exactly the four model artifacts")
    for model, filename in MODEL_FILENAMES.items():
        expected_path = f"embeddings/{filename}"
        _require(models[model].get("filename") == expected_path, f"models.{model}.filename", f"expected {expected_path}")
        expected_trial = None if model == "baseline_iresnet100" else ledger["selected_trials"][model.removesuffix("_selected")]
        _require(
            models[model].get("selected_trial_id") == expected_trial,
            f"models.{model}.selected_trial_id",
            f"expected {expected_trial!r}",
        )
        load_embedding_package(
            Path(package_dir) / expected_path,
            manifest_path=manifest_path,
            expected_dimension=expected_dimension,
            norm_tolerance=norm_tolerance,
        )

    metric_definitions = metadata.get("metric_definitions")
    _require(isinstance(metric_definitions, dict), "metric_definitions", "must be an object")
    _require(
        metric_definitions.get("triplet_accuracy") == TRIPLET_ACCURACY_DEFINITION,
        "metric_definitions.triplet_accuracy",
        "does not match the schema-v1 definition",
    )
    _require(
        metric_definitions.get("similarity_lift") == SIMILARITY_LIFT_DEFINITION,
        "metric_definitions.similarity_lift",
        "does not match the schema-v1 definition",
    )
    embedding_validation = metadata.get("embedding_validation")
    _require(isinstance(embedding_validation, dict), "embedding_validation", "must be an object")
    expected_invariants = {
        "embedding_dimension": int(expected_dimension),
        "embedding_dtype": "float32",
        "index_dtype": "int64",
        "finite": True,
        "unit_norm_tolerance": float(norm_tolerance),
        "index_coverage": "exactly once per manifest split",
    }
    _require(embedding_validation == expected_invariants, "embedding_validation", f"expected {expected_invariants}")

    audit = metadata.get("audit")
    _require(isinstance(audit, dict), "audit", "must be an object")
    _require(audit.get("status") in {"pass", "fail"}, "audit.status", "expected pass or fail")
    _validate_utc_timestamp(audit.get("timestamp"), "audit.timestamp")
    return metadata


def write_handoff_manifest(path, metadata):
    """Write metadata deterministically as UTF-8 JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return path


def read_handoff_manifest(path):
    """Read a handoff manifest, leaving cross-file validation to its validator."""
    with Path(path).open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    _require(isinstance(metadata, dict), "handoff_manifest", "must be an object")
    _require(metadata.get("schema_version") == 1, "schema_version", "expected 1")
    return metadata
