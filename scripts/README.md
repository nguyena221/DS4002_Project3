# Four project scripts

| Script | Commands and purpose |
|---|---|
| explore_data.py | Raw-archive EDA, image counts and plots |
| preprocess.py | detect; estimate-age; filter-classes; prepare; finalize |
| compare_thresholds.py | age; detection; combined; all |
| prepared_dataset.py | PreparedFaces loader used by training code |

## Active dataset

Estimated age >=25, RetinaFace confidence >=0.5, minimum 300/name. Final count: 64,398 images, 100 names. Train 45,016; validation 9,665; test 9,717. Use data/processed/prepared_faces_age25/faces.h5 and output/final_age25/HANDOFF.md. Images are aligned 112x112 RGB float32 CHW, already normalized to [-1,1]. Keep the existing label mapping and split assignments.

## Commands

Run from the repository root using Python 3.11 and requirements-preprocessing-lock.txt. No need to rerun completed preprocessing simply because files were consolidated. All commands protect existing outputs. Use new --output paths where supported, or reproduce the full sequence in a separate checkout with the original data, models and exclusion CSV available.

```powershell
.\.venv\Scripts\python.exe scripts\explore_data.py --help
.\.venv\Scripts\python.exe scripts\preprocess.py --help
.\.venv\Scripts\python.exe scripts\preprocess.py detect --help
.\.venv\Scripts\python.exe scripts\compare_thresholds.py --help
```

To reproduce the recorded pipeline in a clean checkout:

```powershell
.\.venv\Scripts\python.exe scripts\explore_data.py --archive "$env:USERPROFILE\Downloads\Names100Dataset.tar.gz" --output output/eda
.\.venv\Scripts\python.exe scripts\preprocess.py detect --archive "$env:USERPROFILE\Downloads\Names100Dataset.tar.gz"
.\.venv\Scripts\python.exe scripts\preprocess.py estimate-age --archive "$env:USERPROFILE\Downloads\Names100Dataset.tar.gz"
.\.venv\Scripts\python.exe scripts\preprocess.py filter-classes
.\.venv\Scripts\python.exe scripts\preprocess.py prepare --archive "$env:USERPROFILE\Downloads\Names100Dataset.tar.gz"
.\.venv\Scripts\python.exe scripts\preprocess.py finalize
```

The first four preprocessing commands intentionally reproduce the age18 baseline. `finalize` then applies the group-approved age25 filter using those aligned tensors, preserving their labels and original partition assignments. Both the baseline and active tensor files are kept locally. Commands are separate so each stage can be run independently when its prerequisites are available. `finalize` uses fixed recorded inputs/outputs and rejects existing output folders.

## Threshold evidence

The comparisons use the preserved age18 baseline and saved predictions. To rerun without replacing existing reports:

```powershell
.\.venv\Scripts\python.exe scripts\compare_thresholds.py age --output output/recheck_age
.\.venv\Scripts\python.exe scripts\compare_thresholds.py detection --output output/recheck_detection
.\.venv\Scripts\python.exe scripts\compare_thresholds.py combined --output output/recheck_combined
```

Alternatively, `compare_thresholds.py all --output output/recheck_all` writes three subfolders. Individual age/detection modes reuse earlier training-only visual judgments; these are not verified ages or whole-dataset error rates. Combined mode reports exact retention counts without claiming child/cartoon accuracy. No detector inference is rerun. Previously excluded multi-face images are not reconsidered.

## Consolidation and cleanup

Nine old scripts were merged into explore_data.py, preprocess.py and compare_thresholds.py; prepared_dataset.py is unchanged. Original versions and README snapshots are in a local ignored archive under data/archive/script_consolidation. There are four active .py files. Legacy paths were updated in current READMEs; archived documents carry a historical notice. Consolidation verification is recorded in output/final_age25/code_consolidation_verification.json. No image tensor, filtering decision or split was changed.

Older pilots, local models, raw archives and tensors stay out of Git. Only the archived visual-review CSV required by the comparison code is eligible for tracking. Stale Git entries for deleted pilot files were removed in the prior cleanup. No training or evaluation has run.

Current duplicate cleanup: removed Mike_311.png and Steve_550.png from prepared data and all active manifests. data/additional_exclusions.csv makes this reproducible. Removed 139 redundant exact-duplicate copies, keeping one image per group. No exact source/aligned-pixel duplicates remain in the active dataset. Counts concern exact source/aligned-pixel matches, not all photos of the same person.

Final deduplication: 139 redundant exact copies were removed, keeping the alphabetically first filename per transitive source/aligned-pixel duplicate group. The 64,398 retained images have unique source and aligned-pixel hashes. Original retained-image labels and split assignments are unchanged. See output/final_age25/duplicate_removals.csv and duplicate_audit.json. Near duplicates and different photographs of the same person are not comprehensively verified.
