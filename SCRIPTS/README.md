# Script execution order

1. `01_explore_raw_data.py`: streams the original archive, validates decoding,
   counts name categories, measures mean brightness, and writes exploratory outputs.
   Usage from repository root:
   `python SCRIPTS/01_explore_raw_data.py --archive "PATH/TO/Names100Dataset.tar.gz" --output OUTPUT/eda`

Planned next stages, not implemented: a stratified preprocessing pilot, full
screening/alignment with an exclusion manifest, reproducible dataset splitting,
model training, and held-out evaluation. Assign ordered filenames as these
scripts are developed. See `OUTPUT/preprocessing_plan.md` for pending decisions.
