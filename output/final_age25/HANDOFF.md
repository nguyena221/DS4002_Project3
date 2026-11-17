> Current active dataset after exact deduplication: 64,398 images across 100 names; train 45,016, validation 9,665, test 9,717. Age >=25, confidence >=0.5. The Mike/Steve pair and 139 extra exact copies are excluded. See output/final_age25/HANDOFF.md; historical stage counts below are preserved.

# Active preprocessing dataset: estimated age 25+

Group-approved settings: estimated age >=25; RetinaFace confidence >=0.5; minimum 300 images/name. Retained 64,398 images across 100 names; removed 9,581 from the earlier age18 dataset. Mike_311.png and Steve_550.png were explicitly excluded by user instruction because their identical aligned image had conflicting labels. Additional exclusions are recorded in data/additional_exclusions.csv. 139 additional redundant copies were removed by exact source/aligned-pixel hashing; see duplicate_removals.csv. No broader manual review or cartoon detection was performed. These are model-estimated ages; children and artificial faces can remain.

Train 45,016; validation 9,665; test 9,717. Original split assignments were preserved, so percentages are approximately 70/15/15 rather than a new stratification. Every name remains in all three splits. Exact source/pixel duplicates remain within one partition; different photos of the same person are not identity-disjoint. No remaining cross-name exact-duplicate conflicts were found. 139 redundant exact-duplicate copies were removed, keeping one deterministic representative per group; labels were not guessed or changed.

Active data: data/processed/prepared_faces_age25/faces.h5. RGB float32 NCHW, shape (64398,3,112,112), normalized to [-1,1]. Do not normalize again. Labels remain 0-99 with the unchanged label_mapping.json. Use scripts/prepared_dataset.py:

```python
from prepared_dataset import PreparedFaces
train = PreparedFaces('data/processed/prepared_faces_age25/faces.h5', 'train')
image, label = train[0]
train.close()
```

Reproduce from the preserved earlier prepared dataset: `python scripts/preprocess.py finalize`. Existing output paths are protected against overwrite. All source tensors and every saved tensor were hash-verified; loader checks and duplicate separation passed. Earlier age18 files remain as historical inputs, not active training data. No original photos were modified.

The age cutoff changed from MI2's 18 to 25 by group decision after threshold comparisons. Detection stayed at 0.5. Confidence/age filtering does not certify photographic realism or adulthood. No model training or evaluation was performed.
