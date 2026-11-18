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

__all__ = [
    "align_embeddings",
    "build_handoff_manifest",
    "load_embedding_package",
    "read_handoff_manifest",
    "read_manifest",
    "sha256_file",
    "validate_embedding_package",
    "validate_handoff_manifest",
    "validate_manifest_hdf5",
    "validate_trial_ledger",
    "write_handoff_manifest",
    "write_embedding_package",
]
