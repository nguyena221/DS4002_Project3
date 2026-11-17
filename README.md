# Do You Look Like Your Name?

DS4002 Project 3 | Eli Boyden, Zilan Saadi, Annie Nguyen

## Final preprocessing dataset

**64,398 images across 100 first names.** Group-approved settings: estimated age >=25, RetinaFace confidence >=0.5, minimum 300 images per name.

| Split | Images |
|---|---:|
| Train | 45,016 |
| Validation | 9,665 |
| Test | 9,717 |

Faces are aligned with RetinaFace's five landmarks and InsightFace `face_align.norm_crop` at 112x112. Inputs are RGB float32 CHW, normalized to [-1,1]. **Do not normalize again.** Existing split assignments were preserved after filtering, giving approximately 70/15/15; labels remain 0-99. Exact source/prepared-pixel duplicates remain in one split.

- Active local data: `data/processed/prepared_faces_age25/faces.h5` (excluded from Git).
- [Final handoff](output/final_age25/HANDOFF.md): loading, schema, counts, verification and limitations.
- [Final results](output/final_age25): manifests, labels, class counts and duplicate conflicts.
- [Active dataset pointer](data/processed/active_dataset.json).

```python
import sys
sys.path.insert(0, 'scripts')
from prepared_dataset import PreparedFaces
train = PreparedFaces('data/processed/prepared_faces_age25/faces.h5', 'train')
image, label = train[0]
train.close()
```

## Four scripts

| Script | Purpose |
|---|---|
| [explore_data.py](scripts/explore_data.py) | Raw-data EDA and plots |
| [preprocess.py](scripts/preprocess.py) | Separate commands for detection, age estimates, class filtering, alignment/normalization/splitting, and the final age25 update |
| [compare_thresholds.py](scripts/compare_thresholds.py) | Age, detection, combined, or all threshold comparisons |
| [prepared_dataset.py](scripts/prepared_dataset.py) | Read-only training-data loader |

See [scripts/README.md](scripts/README.md) for commands. Use Python 3.11 and the recorded `requirements-preprocessing-lock.txt`. Consolidation changed code organization only; it did not rerun models, alter image tensors, or change splits. Comparison CSVs were reproduced byte-for-byte. Existing outputs are protected against overwrite.

## Results and local history

| Folder | Purpose |
|---|---|
| output/final_age25 | Active results and handoff |
| output/eda | Raw-data plots and summaries |
| output/age_cutoff_comparison | Age18/25/30 experiment |
| output/detection_threshold_comparison | Confidence0.5/0.7/0.9 experiment |
| output/age_detection_comparison | All nine combinations |
| output/face_detection | Full RetinaFace decisions and landmarks |
| output/age_filter, class_filter, prepared_dataset | Historical age18 baseline for reproduction and comparisons |

Older pilots remain locally under data/archive, excluded from Git except the visual-review CSV needed by comparisons. Original script versions and README snapshots are saved in a local ZIP under data/archive/script_consolidation. The obsolete preprocessing plan was archived; stale Git entries for already-deleted pilot files and disposable Python caches were removed. No commit or push was made by the cleanup.

Local models, environments, source archives and image tensors are ignored by Git. The earlier `data/processed/prepared_faces/faces.h5` is still required by `preprocess.py finalize`; keep it for reproduction. Do not train on that age18 baseline by mistake.

## Decisions and limitations

The group selected age25 instead of MI2's age18. RetinaFace is implemented through UniFace; age estimation uses InsightFace buffalo_l. Saved-score experiments establish retention counts, not verified child/cartoon removal accuracy. Children, artificial images and wrong name labels may remain. Different photographs of the same person may cross splits because identity annotations are unavailable.

Mike_311.png and Steve_550.png were removed by user instruction because the identical aligned image had conflicting name labels. No remaining cross-name exact-duplicate conflicts were found; all 139 redundant same-name copies have now been removed, keeping one representative per group. The original 67 exclusions remain in data/user_review_exclusions.csv; the two additional exclusions are recorded in data/additional_exclusions.csv. No training or model evaluation has run.

## Data access and licensing

Original data: [Stanford Name100 repository](https://purl.stanford.edu/tp945cq9122). The original archive remains in Downloads. Obtain image files from the original source; they are not distributed in this Git repository. See [LICENSE.md](LICENSE.md) for project and third-party terms.

Current duplicate cleanup: removed Mike_311.png and Steve_550.png from prepared data and all active manifests. data/additional_exclusions.csv makes this reproducible. Removed 139 redundant exact-duplicate copies, keeping one image per group. No exact source/aligned-pixel duplicates remain in the active dataset. Counts concern exact source/aligned-pixel matches, not all photos of the same person.

Final deduplication: 139 redundant exact copies were removed, keeping the alphabetically first filename per transitive source/aligned-pixel duplicate group. The 64,398 retained images have unique source and aligned-pixel hashes. Original retained-image labels and split assignments are unchanged. See output/final_age25/duplicate_removals.csv and duplicate_audit.json. Near duplicates and different photographs of the same person are not comprehensively verified.
