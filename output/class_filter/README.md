> Current active dataset after exact deduplication: 64,398 images across 100 names; train 45,016, validation 9,665, test 9,717. Age >=25, confidence >=0.5. The Mike/Steve pair and 139 extra exact copies are excluded. See output/final_age25/HANDOFF.md; historical stage counts below are preserved.

> This folder records a completed historical stage or experiment. The active training data uses estimated age >=25 and confidence >=0.5: 64,539 images across 100 names. See output/final_age25/HANDOFF.md. Code was consolidated into four scripts; the commands below use the new names. Original measurements are unchanged.

# Step 3: minimum class size only

Applied the MI2 rule: keep a first name only when it has at least 300 images remaining after the estimated-age >=18 filter. Exactly 300 qualifies.

Input: 73,979 images across 100 names. Retained: 73,979 images across 100 names. Excluded: 0 images across 0 names. Smallest input class: 576; largest: 781.

No source photographs or age results were changed. No detector/age model reruns, image resizing, training normalization, or dataset splits were performed. Stop here for user confirmation.

`name_counts.csv` provides every name and its eligibility. `class_filter_manifest.csv` records the decision for each age-retained image. `retained_manifest.csv` is the input list for the next authorized step. `summary.json` records totals, settings, verification, and the source manifest hash.

Adult eligibility is based on estimated age. The 300-image minimum does not make classes equally sized.
