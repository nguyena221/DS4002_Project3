# `artifacts`

These artifacts, produced during the [`training`](../../scripts/training/README.md) phase, are input files that are used by [`eval`](../../scripts/eval/README.md) to produce one csv file containing the final accuracy, difference from the baseline, and a statistical confidence interval inidcating if any of our selected fine-tuned models show improvements over the baseline un-fine-tuned IResNet-100. Files produced/used include:

- `data/processed/manifest.csv`

File listing the image names, h5 index (a unique number per image), first names, label (from 0-99 corresponding to each name), estimated age, det score, and sha hash for each of the ~64,000 images that survived pre-processing. Code that produced this artifact can be found in [`preprocess`](../../scripts/preprocess/README.md).

- `embeddings/*.npz`

These files contain all of the embeddings for each image in the validation and test data. Each file corresponds to one of the 4 models we trained (baseline un-fine-tuned iresnet100, fine-tuned triplet margin loss, fine-tuned arcface loss, and fine-tuned supcon loss).

Here is the full shape of each `embeddings/*.npz` file:

| Key | Type/shape | Meaning |
|---|---|---|
| `validation_h5_index` | `int64`, `(9660,)` | Original image IDs, in embedding-row order |
| `validation_embeddings` | `float32`, `(9660, 512)` | Unit-normalized validation vectors |
| `test_h5_index` | `int64`, `(9660,)` | Original image IDs, in embedding-row order |
| `test_embeddings` | `float32`, `(9660, 512)` | Unit-normalized test vectors |

- `trial_results.csv`

This file contains the results of each fine-tuning run for all 10 models selected over during validation. For all 10 trails this includes a trial_id, the best validation loss, the hyperparameters used, and other less relevant information.

- `handoff_manifest.json`

This file contains lots of metadata about the data used/produced in training. Most importantly, under "models" the file includes all of the trial_id's (the same trial_id found in `trial_results.csv`) of the selected model (for each loss function).
