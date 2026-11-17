> Current active dataset after exact deduplication: 64,398 images across 100 names; train 45,016, validation 9,665, test 9,717. Age >=25, confidence >=0.5. The Mike/Steve pair and 139 extra exact copies are excluded. See output/final_age25/HANDOFF.md; historical stage counts below are preserved.

> This folder records a completed historical stage or experiment. The active training data uses estimated age >=25 and confidence >=0.5: 64,539 images across 100 names. See output/final_age25/HANDOFF.md. Code was consolidated into four scripts; the commands below use the new names. Original measurements are unchanged.

> Historical age18 baseline, superseded for active training. See ../final_age25/HANDOFF.md for the approved age25 dataset.

# Prepared dataset and fixed splits

Prepared all 73,979 Step 3 images, with no further removals. Five-point RetinaFace landmarks were mapped to the standard ArcFace template at 112x112. Images are RGB, float32 CHW, normalized by x/127.5-1. Saved once in a losslessly compressed HDF5 file, not as thousands of duplicate photo/tensor files.

Splits: training 51,785, validation 11,097, test 11,097. Two-stage name-stratified splitting used seed 4002; same-name swaps kept 149 exact duplicate groups within a single split while preserving every per-name split count. All 100 names occur in each split.

Every saved tensor was checked for shape, dtype, finite values, [-1,1] range, and exact SHA-256 agreement with its prepared tensor. Normalization was checked to round-trip to the aligned RGB pixels. Filename/index/label links and disjoint split/source/pixel hashes were verified. Exact duplicates are not the same as identity labels: distinct photos of the same person may remain across splits. No claim of verified adulthood is made.

Data: data/processed/prepared_faces/faces.h5. Use scripts/prepared_dataset.py. All decisions and source hashes remain traceable in split_manifest.csv. Per-split manifests, per-name counts, duplicate groups, label mapping, and summary.json are included here. alignment_preview.jpg is a small local QA view and is excluded from Git.

No training or evaluation has been run. Training augmentation and loss-specific sampling are not implemented in this preprocessing stage. Preserve these split assignments for comparisons across loss functions.
