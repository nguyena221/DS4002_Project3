> Current active dataset after exact deduplication: 64,398 images across 100 names; train 45,016, validation 9,665, test 9,717. Age >=25, confidence >=0.5. The Mike/Steve pair and 139 extra exact copies are excluded. See output/final_age25/HANDOFF.md; historical stage counts below are preserved.

> This folder records a completed historical stage or experiment. The active training data uses estimated age >=25 and confidence >=0.5: 64,539 images across 100 names. See output/final_age25/HANDOFF.md. Code was consolidated into four scripts; the commands below use the new names. Original measurements are unchanged.

# Detection confidence comparison on updated dataset

Compares saved RetinaFace confidence scores at 0.5, 0.7, and 0.9. Age cutoff stays at 18; original image preparation, manual exclusions, and train/validation/test assignments remain unchanged. Reproduce with `python scripts/compare_thresholds.py detection` from a checkout without this output folder. Existing output is protected against overwrite.

| Confidence cutoff | Images retained | Additional removed | Names meeting 300 minimum | Questionable among reviewed retained |
|---:|---:|---:|---:|---:|
| 0.5 | 73,979 | 0 | 100 | 58/125 (46.4%) |
| 0.7 | 73,948 | 31 | 100 | 58/125 (46.4%) |
| 0.9 | 73,893 | 86 | 100 | 58/125 (46.4%) |

Counts apply to the current prepared dataset, not a new full 80,000-image detection run. This post-filter comparison does not reconsider previously excluded multi-face images that could become single-face at another cutoff. Model weights, input size and NMS are fixed. No model inference was rerun.

Visual diagnostics reuse earlier assistant judgments matched by original file hash and restricted to the current training split. These are neither verified age labels nor a fresh independent review. Questionable includes uncertain ages and unassessable images. Percentages are descriptive of this small reviewed subset and do not estimate whole-dataset error rates. Confidence measures detector confidence, not adulthood, realism, or correct identity.

No threshold has been adopted. Class-minimum effects are hypothetical. CSV files record summary results, per-name counts, image-level experimental membership, and training-only visual judgments. summary.json records source hashes and limitations. Assertions check baseline retention, nested thresholds, matching source hashes, and unchanged input manifests. No detection precision/recall can be established without annotated ground truth.
