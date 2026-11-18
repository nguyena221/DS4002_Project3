# Scripts

Contains all scripts used to prepreprocess, preprocess, postpreprocess, processs and postprocess the data. Run from the repository root with Python 3.11 and `requirements.txt`.

| Script | Purpose |
|---|---|
| `explore_data.py` | EDA on the raw archive; writes figures to `output/eda/` |
| `preprocess.py` | Raw archive to `data/processed/faces.h5` and `manifest.csv` |
| `prepared_dataset.py` | `PreparedFaces` loader for one split (PyTorch-compatible, no torch needed) |

## `preprocess.py` steps

One function per step, run in order:

1. `detect_face`: RetinaFace (UniFace), keep images with exactly one face (confidence >= 0.5)
2. `estimate_age`: InsightFace `buffalo_l`, keep estimated age >= 25
3. `align_face`: five-point alignment to 112x112
4. `remove_duplicates`: drop exact copies (source or aligned pixels)
5. `filter_classes`: keep names with at least 300 images
6. `split_dataset`: stratified 70/15/15 split, seed 4002

Manually excluded images (`data/exclusions.csv`) are removed first. Setting variables are at the top of the file. Needs the `buffalo_l` age model in `~/.insightface/models/buffalo_l/`.
