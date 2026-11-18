# Data

| Path | Contents | In Git |
|---|---|---|
| `raw/Names100Dataset.tar.gz` | Original archive ([source](https://purl.stanford.edu/tp945cq9122)) | no |
| `exclusions.csv` | 69 images the group removed by hand (SHA-256 hash, reason) | yes |
| `processed/faces.h5` | Final dataset, 64,398 aligned faces | no |
| `processed/manifest.csv` | One row per image: HDF5 index, filename, name, label, split, estimated age, detection score, source hash | yes |

## `faces.h5`

| Dataset | Shape / type |
|---|---|
| `images` | (64398, 3, 112, 112) float32, RGB, normalized to [-1, 1] |
| `labels` | int64, 0-99 (sorted by first name) |
| `split` | 0 train, 1 validation, 2 test |
| `split_indices/{train,validation,test}` | indices into `images` |
| `filenames`, `class_names` | strings |

Load with `scripts/prepared_dataset.py`. The data is for course use and subject to the dataset's terms.
