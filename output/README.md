# Results

Start with final_age25/HANDOFF.md. Active dataset: 64,398 images, 100 names; train 45,016, validation 9,665, test 9,717. Age >=25, RetinaFace >=0.5, minimum 300/name; aligned 112x112 RGB float32 CHW, normalized [-1,1].

| Folder | Contents |
|---|---|
| final_age25 | Active manifests, labels, counts, handoff and verification |
| eda | Original raw-data plots and captions |
| age_cutoff_comparison | Age18/25/30 experiment |
| detection_threshold_comparison | Confidence0.5/0.7/0.9 experiment |
| age_detection_comparison | All nine combinations |
| face_detection | Original full face decisions and landmarks |
| age_filter, class_filter, prepared_dataset | Historical age18 baseline for reproduction |

Use scripts/explore_data.py for EDA, scripts/preprocess.py with a stage command for preprocessing, scripts/compare_thresholds.py with a mode for experiments, and scripts/prepared_dataset.py for loading. See scripts/README.md for examples. Existing results were not recalculated or replaced during consolidation; comparison CSVs were independently reproduced byte-for-byte. code_consolidation_verification.json under final_age25 records the checks.

Comparisons reuse saved predictions; retained-image counts do not establish whole-dataset child/cartoon removal accuracy. Earlier pilots and the obsolete plan are archived locally in data/archive, not active results. Prior cleanup removed stale Git entries and caches, while original data and evidence were preserved.

Current duplicate cleanup: removed Mike_311.png and Steve_550.png from prepared data and all active manifests. data/additional_exclusions.csv makes this reproducible. Removed 139 redundant exact-duplicate copies, keeping one image per group. No exact source/aligned-pixel duplicates remain in the active dataset. Counts concern exact source/aligned-pixel matches, not all photos of the same person.

Final deduplication: 139 redundant exact copies were removed, keeping the alphabetically first filename per transitive source/aligned-pixel duplicate group. The 64,398 retained images have unique source and aligned-pixel hashes. Original retained-image labels and split assignments are unchanged. See output/final_age25/duplicate_removals.csv and duplicate_audit.json. Near duplicates and different photographs of the same person are not comprehensively verified.
