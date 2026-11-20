"""Fresh-environment evaluation of a copied face-name handoff package."""

from pathlib import Path

import numpy as np

from scripts.face_name_metrics import (
    blocked_triplet_accuracy,
    direct_triplet_accuracy,
    paired_bootstrap_statistics,
    stratified_bootstrap_multiplicities,
)
from .training import named_child_seed

from .handoff import (
    MODEL_FILENAMES,
    align_embeddings,
    load_embedding_package,
    read_handoff_manifest,
    read_manifest,
    validate_handoff_manifest,
)


def handoff_package_readiness(package_dir):
    """Report whether generated production inputs are present."""
    package_dir = Path(package_dir)
    expected = (
        "trial_results.csv",
        "handoff_manifest.json",
        *(f"embeddings/{filename}" for filename in MODEL_FILENAMES.values()),
    )
    missing = [
        relative
        for relative in expected
        if not (package_dir / relative).is_file()
    ]
    return {
        "status": "ready" if not missing else "pending",
        "missing": missing,
    }


def evaluate_handoff_package(
    package_dir,
    *,
    manifest_path,
    trial_results_path=None,
    handoff_manifest_path=None,
    hdf5_path=None,
    expected_dimension=512,
    norm_tolerance=1e-5,
    block_size=1024,
    bootstrap_replicates=2000,
    direct_oracle=False,
):
    """Validate and evaluate only the shared metadata and embedding package."""
    package_dir = Path(package_dir)
    trial_results_path = (
        Path(trial_results_path)
        if trial_results_path is not None
        else package_dir / "trial_results.csv"
    )
    handoff_manifest_path = (
        Path(handoff_manifest_path)
        if handoff_manifest_path is not None
        else package_dir / "handoff_manifest.json"
    )
    metadata = read_handoff_manifest(handoff_manifest_path)
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
    split_rows = {
        split: [row for row in rows if row["split"] == split]
        for split in ("validation", "test")
    }
    split_embeddings = {}
    model_results = {}
    for model, filename in MODEL_FILENAMES.items():
        package = load_embedding_package(
            package_dir / "embeddings" / filename,
            manifest_path=manifest_path,
            expected_dimension=expected_dimension,
            norm_tolerance=norm_tolerance,
        )
        model_results[model] = {"splits": {}}
        split_embeddings[model] = {}
        for split, target_rows in split_rows.items():
            target_indices = np.asarray(
                [row["h5_index"] for row in target_rows], dtype=np.int64
            )
            first_names = np.asarray(
                [row["first_name"] for row in target_rows], dtype=object
            )
            aligned = align_embeddings(
                package[f"{split}_h5_index"],
                package[f"{split}_embeddings"],
                target_indices,
            )
            blocked = float(
                blocked_triplet_accuracy(
                    aligned,
                    target_indices,
                    first_names,
                    block_size=block_size,
                )
            )
            split_result = {"blocked": blocked}
            if direct_oracle:
                direct = float(
                    direct_triplet_accuracy(
                        aligned, target_indices, first_names
                    )
                )
                if not np.isclose(blocked, direct, rtol=0.0, atol=1e-12):
                    raise ValueError(
                        f"acceptance.metric.{model}.{split}: "
                        f"blocked={blocked}, direct={direct}"
                    )
                split_result["direct"] = direct
            split_embeddings[model][split] = aligned
            model_results[model]["splits"][split] = split_result

    baseline_model = "baseline_iresnet100"
    for model in MODEL_FILENAMES:
        for split in split_rows:
            accuracy = model_results[model]["splits"][split]["blocked"]
            baseline = model_results[baseline_model]["splits"][split]["blocked"]
            model_results[model]["splits"][split]["similarity_lift"] = (
                accuracy - baseline
            )

    draw_weights = {
        split: stratified_bootstrap_multiplicities(
            np.asarray(
                [row["first_name"] for row in target_rows], dtype=object
            ),
            replicates=bootstrap_replicates,
            seed=named_child_seed(
                metadata["root_seed"],
                "bootstrap",
                split,
            ),
        )
        for split, target_rows in split_rows.items()
    }
    bootstrap = {}
    for model in MODEL_FILENAMES:
        if model == baseline_model:
            continue
        bootstrap[model] = {}
        for split, target_rows in split_rows.items():
            statistics = paired_bootstrap_statistics(
                split_embeddings[baseline_model][split],
                split_embeddings[model][split],
                np.asarray(
                    [row["h5_index"] for row in target_rows], dtype=np.int64
                ),
                np.asarray(
                    [row["first_name"] for row in target_rows], dtype=object
                ),
                replicates=bootstrap_replicates,
                draw_weights=draw_weights[split],
            )
            bootstrap[model][split] = {
                "confidence_interval": (
                    statistics["confidence_interval"].astype(float).tolist()
                ),
                "baseline_triplet_accuracies": (
                    statistics["baseline_triplet_accuracies"]
                    .astype(float)
                    .tolist()
                ),
                "comparison_triplet_accuracies": (
                    statistics["comparison_triplet_accuracies"]
                    .astype(float)
                    .tolist()
                ),
                "similarity_lifts": (
                    statistics["similarity_lifts"].astype(float).tolist()
                ),
            }

    return {
        "manifest": metadata,
        "models": model_results,
        "bootstrap": bootstrap,
    }
