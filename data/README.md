# Data

Active local tensors: processed/prepared_faces_age25/faces.h5. 64,398 images, 100 names; train 45,016, validation 9,665, test 9,717. Settings: estimated age >=25, RetinaFace >=0.5, minimum 300/name. Faces are landmark-aligned 112x112 RGB float32 CHW and normalized to [-1,1]. Do not normalize again.

processed/active_dataset.json identifies the current file. Use scripts/prepared_dataset.py to load it; output/final_age25/HANDOFF.md describes schema and labels. scripts/preprocess.py finalize reproduces the update from the historical processed/prepared_faces/faces.h5; keep that baseline for reproduction, not active training.

user_review_exclusions.csv preserves 67 explicit decisions. The original archive is in Downloads; source: https://purl.stanford.edu/tp945cq9122. Images, models and tensor files are excluded from Git. Ages are estimates; this is not confirmed adult-only photographic data.

archive/ contains historical experiments, the obsolete plan, cleanup records, and original scripts/README snapshots from consolidation. These are local history and ignored by Git, except the exact visual-review CSV used by scripts/compare_thresholds.py age and detection. Existing comparison evidence remains intact. Four active scripts now cover EDA, preprocessing, comparisons, and loading; see scripts/README.md.

Current duplicate cleanup: removed Mike_311.png and Steve_550.png from prepared data and all active manifests. data/additional_exclusions.csv makes this reproducible. Removed 139 redundant exact-duplicate copies, keeping one image per group. No exact source/aligned-pixel duplicates remain in the active dataset. Counts concern exact source/aligned-pixel matches, not all photos of the same person.

Final deduplication: 139 redundant exact copies were removed, keeping the alphabetically first filename per transitive source/aligned-pixel duplicate group. The 64,398 retained images have unique source and aligned-pixel hashes. Original retained-image labels and split assignments are unchanged. See output/final_age25/duplicate_removals.csv and duplicate_audit.json. Near duplicates and different photographs of the same person are not comprehensively verified.
