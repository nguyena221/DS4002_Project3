"""Independent, weights-free audit of a face-name evaluation handoff."""

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from scripts.face_name_metrics import blocked_triplet_accuracy, direct_triplet_accuracy

from .handoff import (
    MODEL_FILENAMES,
    align_embeddings,
    load_embedding_package,
    read_handoff_manifest,
    read_manifest,
    validate_handoff_manifest,
    write_handoff_manifest,
)


def _utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _sample_validation_rows(rows, images_per_class):
    by_name = {}
    for row in rows:
        if row["split"] == "validation":
            by_name.setdefault(row["first_name"], []).append(row)
    sampled = []
    for first_name in sorted(by_name):
        sampled.extend(
            sorted(by_name[first_name], key=lambda row: row["h5_index"])[
                :images_per_class
            ]
        )
    sampled_names = {row["first_name"] for row in sampled}
    positive_name = any(
        sum(row["first_name"] == name for row in sampled) >= 2
        for name in sampled_names
    )
    if len(sampled_names) < 2 or not positive_name:
        raise ValueError(
            "audit.metric_sample: expected at least two classes and one same-name pair"
        )
    return sampled


def audit_handoff_package(
    package_dir,
    *,
    manifest_path,
    hdf5_path,
    trial_results_path,
    handoff_manifest_path=None,
    expected_dimension=512,
    norm_tolerance=1e-5,
    images_per_class=2,
    audit_timestamp=None,
):
    """Reload, independently check, and mark a completed handoff package.

    The check never loads model weights. It aligns persisted validation
    embeddings by ``h5_index`` and verifies a small deterministic sample twice:
    once with the blocked implementation and once with direct enumeration.
    Any audit failure is persisted to the manifest before it is raised.
    """
    package_dir = Path(package_dir)
    handoff_manifest_path = (
        Path(handoff_manifest_path)
        if handoff_manifest_path is not None
        else package_dir / "handoff_manifest.json"
    )
    timestamp = audit_timestamp or _utc_now()
    metadata = read_handoff_manifest(handoff_manifest_path)
    audited = deepcopy(metadata)

    try:
        validate_handoff_manifest(
            metadata,
            package_dir,
            manifest_path=manifest_path,
            hdf5_path=hdf5_path,
            trial_results_path=trial_results_path,
            expected_dimension=expected_dimension,
            norm_tolerance=norm_tolerance,
        )
        rows = read_manifest(manifest_path)
        validation_rows = [
            row for row in rows if row["split"] == "validation"
        ]
        sample_rows = _sample_validation_rows(rows, images_per_class)
        sample_indices = np.array(
            [row["h5_index"] for row in sample_rows], dtype=np.int64
        )
        sample_names = np.array(
            [row["first_name"] for row in sample_rows], dtype=object
        )
        target_indices = np.array(
            [row["h5_index"] for row in validation_rows], dtype=np.int64
        )
        positions = {
            row["h5_index"]: position
            for position, row in enumerate(validation_rows)
        }
        metric_results = {}
        for model, filename in MODEL_FILENAMES.items():
            package = load_embedding_package(
                package_dir / "embeddings" / filename,
                manifest_path=manifest_path,
                expected_dimension=expected_dimension,
                norm_tolerance=norm_tolerance,
            )
            aligned = align_embeddings(
                package["validation_h5_index"],
                package["validation_embeddings"],
                target_indices,
            )
            sample_embeddings = aligned[
                [positions[int(index)] for index in sample_indices]
            ]
            direct = direct_triplet_accuracy(
                sample_embeddings, sample_indices, sample_names
            )
            blocked = blocked_triplet_accuracy(
                sample_embeddings,
                sample_indices,
                sample_names,
                block_size=min(32, len(sample_indices)),
            )
            if not np.isclose(blocked, direct, rtol=0.0, atol=1e-12):
                raise ValueError(
                    f"audit.metric.{model}: blocked={blocked}, direct={direct}"
                )
            metric_results[model] = float(direct)

        audited["audit"] = {
            "status": "pass",
            "timestamp": timestamp,
            "metric": {
                "split": "validation",
                "images_per_class": int(images_per_class),
                "sample_id_order": [int(index) for index in sample_indices],
                "models": metric_results,
                "oracle": "direct ordered-triplet enumeration",
            },
        }
    except Exception as error:
        audited["audit"] = {
            "status": "fail",
            "timestamp": timestamp,
            "error": str(error),
        }
        write_handoff_manifest(handoff_manifest_path, audited)
        raise

    write_handoff_manifest(handoff_manifest_path, audited)
    return validate_handoff_manifest(
        audited,
        package_dir,
        manifest_path=manifest_path,
        hdf5_path=hdf5_path,
        trial_results_path=trial_results_path,
        expected_dimension=expected_dimension,
        norm_tolerance=norm_tolerance,
    )
