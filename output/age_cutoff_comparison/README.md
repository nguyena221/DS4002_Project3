> Current active dataset after exact deduplication: 64,398 images across 100 names; train 45,016, validation 9,665, test 9,717. Age >=25, confidence >=0.5. The Mike/Steve pair and 139 extra exact copies are excluded. See output/final_age25/HANDOFF.md; historical stage counts below are preserved.

> This folder records a completed historical stage or experiment. The active training data uses estimated age >=25 and confidence >=0.5: 64,539 images across 100 names. See output/final_age25/HANDOFF.md. Code was consolidated into four scripts; the commands below use the new names. Original measurements are unchanged.

# Updated dataset: age-cutoff comparison only

Compared the agreed estimated-age cutoffs 18, 25, and 30 using the saved buffalo_l age estimates derived from the actual RetinaFace bounding boxes. No detector settings or model weights were changed and no model inference was rerun. Current HDF5 data, split manifests, and filtering decisions remain unchanged.

## Exact retention counts

The comparison starts with the current 73,979 prepared images, already estimated >=18. The upstream 5,463 estimated-under-18 images remain excluded and are not reintroduced.

| Estimated-age cutoff | Images retained | Additional images removed | Names meeting >=300 | Images after >=300 rule |
|---:|---:|---:|---:|---:|
| 18 | 73,979 | 0 | 100 | 73,979 |
| 25 | 64,539 | 9,440 | 100 | 64,539 |
| 30 | 47,115 | 26,864 | 98 | 46,567 |

Class-minimum effects are hypothetical, reported separately for each cutoff. No new name categories have actually been removed.

## Existing visual quality check: training images only

Reused 125 earlier assistant judgments that match current training images by filename and original hash. Judgments describe original image content, so remain applicable even though the detector/age estimates changed. Current RetinaFace-derived age estimates were used for this comparison, not the old SCRFD ages. Validation/test images were excluded from these quality diagnostics. This is a development check, not a fresh independent validation or verified age ground truth.

| Cutoff | Questionable / retained reviewed | Flag rate | Child-appearing kept | Artificial/stylized kept | Adult-appearing kept |
|---:|---:|---:|---:|---:|---:|
| 18 | 58/125 | 46.4% | 28 | 5 | 67/67 |
| 25 | 40/101 | 39.6% | 19 | 3 | 61/67 |
| 30 | 26/75 | 34.67% | 11 | 3 | 49/67 |

Questionable includes age uncertainty and unassessable images; it does not mean every flagged image is definitely ineligible. Higher cutoffs may remove true adults, especially younger adults. A smaller retained set is not automatically a more accurate one. Do not reuse old SCRFD-based threshold percentages as current-pipeline results.

## What remains unchanged

- Current prepared dataset: 73,979 images, estimated-age cutoff 18.
- Existing train/validation/test assignments and label mapping.
- Original source photographs and confirmed manual exclusions.
- RetinaFace confidence and other detection settings.

No new cutoff has been adopted. Before adopting one, discuss the data/quality tradeoff and arrange fresh validation. A changed cutoff requires updating the active manifests and checking the split/class distributions; this experiment does not silently replace them. Detection-threshold testing has not been performed in this step.

## Files

- age_cutoff_comparison.csv: exact counts and training-only visual diagnostics.
- name_counts_by_cutoff.csv: all 100 name counts for every cutoff.
- experimental_membership.csv: flags mapped to current HDF5 indices and current splits; these are experimental filters, not new splits.
- training_visual_audit_comparison.csv: image-level development judgments and updated threshold outcomes.
- summary.json: provenance, assumptions, and source-file hashes.
