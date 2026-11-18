"""Public utilities for the face-name metric-learning handoff."""

from .handoff import (
    align_embeddings,
    build_handoff_manifest,
    load_embedding_package,
    read_handoff_manifest,
    read_manifest,
    sha256_file,
    validate_embedding_package,
    validate_handoff_manifest,
    validate_manifest_hdf5,
    validate_trial_ledger,
    write_handoff_manifest,
    write_embedding_package,
)
from .training import (
    collect_environment_record,
    export_baseline_embeddings,
    load_pretrained_iresnet100,
    write_environment_record,
)

__all__ = [
    "align_embeddings",
    "build_handoff_manifest",
    "collect_environment_record",
    "export_baseline_embeddings",
    "load_embedding_package",
    "load_pretrained_iresnet100",
    "read_handoff_manifest",
    "read_manifest",
    "sha256_file",
    "validate_embedding_package",
    "validate_handoff_manifest",
    "validate_manifest_hdf5",
    "validate_trial_ledger",
    "write_handoff_manifest",
    "write_environment_record",
    "write_embedding_package",
]
