> Current active dataset after exact deduplication: 64,398 images across 100 names; train 45,016, validation 9,665, test 9,717. Age >=25, confidence >=0.5. The Mike/Steve pair and 139 extra exact copies are excluded. See output/final_age25/HANDOFF.md; historical stage counts below are preserved.

> This folder records a completed historical stage or experiment. The active training data uses estimated age >=25 and confidence >=0.5: 64,539 images across 100 names. See output/final_age25/HANDOFF.md. Code was consolidated into four scripts; the commands below use the new names. Original measurements are unchanged.

# Step 2: estimated-age filtering only

Processed all 79,442 images retained after face detection and prior user exclusions. The buffalo_l age model estimated age >=18 for 73,979 and under 18 for 5,463. Original images and Step 1 outputs are unchanged.

Uses InsightFace's `Attribute.get`, the age module underlying `FaceAnalysis`, with the existing RetinaFace bounding boxes. No detector was rerun. The joint model's gender output is discarded and not saved. The model's internal 96x96 inference crop is temporary and is not the later 112x112 training input.

Age cutoff fixed at 18 as specified in MI2; no age or detector threshold comparisons. No class-size filtering, saved training crops/tensors, normalization for training, or train/validation/test splits. Stop here for user confirmation.

These are estimated ages, not verified adulthood. The previously observed age-model errors remain a limitation. A changed face detector changes boxes supplied to the age module; do not treat earlier pilot error rates as a direct measurement of this new stage.

Files: `age_filter_manifest.csv` records every input and age decision; `name_counts.csv` gives per-name counts without dropping any names; `estimated_age_counts.csv` gives aggregate estimated-age frequencies; `summary.json` records settings, counts, versions, and model hash. Images remain in the original archive.
