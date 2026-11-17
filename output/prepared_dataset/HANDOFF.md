> Current active dataset after exact deduplication: 64,398 images across 100 names; train 45,016, validation 9,665, test 9,717. Age >=25, confidence >=0.5. The Mike/Steve pair and 139 extra exact copies are excluded. See output/final_age25/HANDOFF.md; historical stage counts below are preserved.

> This folder records a completed historical stage or experiment. The active training data uses estimated age >=25 and confidence >=0.5: 64,539 images across 100 names. See output/final_age25/HANDOFF.md. Code was consolidated into four scripts; the commands below use the new names. Original measurements are unchanged.

> Historical age18 baseline, superseded for active training. See ../final_age25/HANDOFF.md for the approved age25 dataset.

# Prepared dataset handoff

All three requested steps are complete: aligned 112x112 RGB faces; float32 CHW normalization to [-1,1]; name-stratified fixed 70/15/15 splits. No additional filtering or training was performed.

| Split | Images |
|---|---:|
| Train | 51,785 |
| Validation | 11,097 |
| Test | 11,097 |
| Total | 73,979 |

All 100 names occur in each split. Seed: 4002 (second split uses 4003). Integer rounding gives approximately 70/15/15. 149 groups of exact source/prepared-pixel duplicates were kept within a single split. Per-name split counts were preserved by swaps. Different photographs of the same person may still cross splits because identity annotations are unavailable.

## Data and loading

File: `data/processed/prepared_faces/faces.h5` (5.21 GB). This one file stores the actual prepared RGB images as normalized float32 arrays, with lossless gzip/shuffle compression and chunk checksums. It is ignored by Git. It replaces the need to store separate unnormalized crops and tens of thousands of .npy files.

```python
import sys
sys.path.insert(0, 'scripts')
from prepared_dataset import PreparedFaces

train = PreparedFaces('data/processed/prepared_faces/faces.h5', split='train')
image, label = train[0]
print(image.shape, image.dtype, label)  # (3,112,112), float32, int
train.close()
```

The map-style dataset can be passed to a PyTorch DataLoader; its default collation converts NumPy arrays to tensors. Each worker opens its own read-only HDF5 handle. PyTorch is part of the later training environment and was not installed or used to train a model here. Inputs are already normalized: do not apply another ToTensor/Normalize pipeline to them. Apply any agreed training-only augmentation separately.

## HDF5 schema

| Dataset | Meaning |
|---|---|
| images | N x 3 x 112 x 112 float32 RGB values in [-1,1] |
| labels | N integer name labels, 0-99 |
| filenames | N original image filenames |
| class_names | 100 names in integer-label order |
| split | N IDs: 0=train, 1=validation, 2=test |
| split_indices/train | Global HDF5 row indices for training |
| split_indices/validation | Global row indices for validation |
| split_indices/test | Global row indices for test |
| normalized_sha256 | Per-image normalized-tensor hashes |

## Documentation

- `split_manifest.csv`: complete row-to-filename/label/split mapping, source and prepared-image hashes, estimated age, and duplicate adjustments.
- `train_manifest.csv`, `validation_manifest.csv`, `test_manifest.csv`: per-split mappings.
- `split_counts_by_name.csv`: every name's counts in all three splits.
- `label_mapping.json`: name-to-integer mapping.
- `duplicate_groups.csv`: duplicate membership and final partition.
- `label_conflicts.csv`: one identical prepared-crop pair labeled Mike and Steve; both are in training. Original labels were preserved and need group review.
- `summary.json`: preprocessing, storage, split, and verification settings.
- `alignment_preview.jpg`: a 16-image local visual QA sheet, excluded from Git.
- `loader_verification.json`: loader checks and whole-file hash.

Whole HDF5 SHA-256: `1ee7361d6f1eb1cf12b74a9d3f4bdaac5090305568f195a71ee2351377267c08`.

## Reproduction

Use Python 3.11. The environment is recorded in `requirements-preprocessing-lock.txt`. On Windows, building InsightFace may require Microsoft C++ Build Tools. This project's environment uses opencv-python-headless as the cv2 provider; UniFace was installed without replacing that provider. To restore the exact listed packages, install the complete lock file with `python -m pip install --no-deps -r requirements-preprocessing-lock.txt`.

After reproducing face detection, age >=18 filtering, and the >=300 class check:

```powershell
.\.venv\Scripts\python.exe scripts\preprocess.py prepare --archive "$env:USERPROFILE\Downloads\Names100Dataset.tar.gz"
```

Existing output paths are never overwritten. For a repeat, pass new `--data-output` and `--output` paths. The original source archive remains unchanged. Labels, filenames, and source-image hashes remain traceable.

## Limits

Eligibility is based on model-estimated age; these are not confirmed adults. No new manual cleanup, threshold experiments, or identity verification was performed here. These fixed splits should be reused across the three planned loss functions. No training or final evaluation has been started.

One duplicate prepared-crop group has conflicting first-name labels: Mike_311.png and Steve_550.png, both in training. This prevents the pair crossing splits but does not resolve label correctness. The group should review it before final training; no relabeling or removal was performed without supporting information. The local alignment preview checks geometry, not adult eligibility or photographic authenticity.
