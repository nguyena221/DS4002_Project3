> Current active dataset after exact deduplication: 64,398 images across 100 names; train 45,016, validation 9,665, test 9,717. Age >=25, confidence >=0.5. The Mike/Steve pair and 139 extra exact copies are excluded. See output/final_age25/HANDOFF.md; historical stage counts below are preserved.

> This folder records a completed historical stage or experiment. The active training data uses estimated age >=25 and confidence >=0.5: 64,539 images across 100 names. See output/final_age25/HANDOFF.md. Code was consolidated into four scripts; the commands below use the new names. Original measurements are unchanged.

# Age and detection confidence comparison

All nine combinations of saved estimated-age cutoffs 18/25/30 and RetinaFace confidence cutoffs 0.5/0.7/0.9 on the current prepared dataset. Reproduce with `python scripts/compare_thresholds.py combined` when this output folder does not already exist. Active data, manual exclusions, labels and splits are unchanged.

| Age | Detection | Kept before class minimum | Extra removed versus detection 0.5 at same age | Kept after class minimum | Names retained |
|---:|---:|---:|---:|---:|---:|
| 18 | 0.5 | 73,979 | 0 | 73,979 | 100 |
| 18 | 0.7 | 73,948 | 31 | 73,948 | 100 |
| 18 | 0.9 | 73,893 | 86 | 73,893 | 100 |
| 25 | 0.5 | 64,539 | 0 | 64,539 | 100 |
| 25 | 0.7 | 64,511 | 28 | 64,511 | 100 |
| 25 | 0.9 | 64,462 | 77 | 64,462 | 100 |
| 30 | 0.5 | 47,115 | 0 | 46,567 | 98 |
| 30 | 0.7 | 47,097 | 18 | 46,549 | 98 |
| 30 | 0.9 | 47,057 | 58 | 46,511 | 98 |

The class minimum is 300 images per name, calculated separately for every combination. These are hypothetical filters, not newly adopted settings. Saved model estimates are reused; detector inference is not rerun. Previously excluded multi-face images are not reconsidered, so this is not a full raw-data detector rerun.

These counts cannot tell how many children or artificial characters remain, nor whether removed images were unwanted. No whole-dataset visual labels are available. Higher confidence is not evidence of adulthood or photographic realism. No new visual validation or accuracy measurement was performed. Source hashes and assertions verify matching input records, unchanged manifests and nested retention counts.
