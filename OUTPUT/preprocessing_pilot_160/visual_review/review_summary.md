# Pilot visual review

## Scope and method

Reviewed 171 unique images using 11 local contact sheets. Selection included 50 randomly sampled retained images (seed 4002), all 94 retained images with estimated ages 18-24, 20 sampled exclusions with estimated ages under 18 (prioritizing estimates 15-17), and all 14 detector exclusions. Seven random retained images also belonged to the borderline group; groups overlap. Original images were reviewed alongside saved aligned crops where available. Contact-sheet labels omitted filenames, selection groups, and estimated ages during inspection.

This is an assistant visual assessment, NOT independent human review or age ground truth. Categories describe apparent age/content only. All cases needing confirmation have a blank second-reviewer column. No original images, screening decisions, age cutoff, class thresholds, or splits were changed.

## Findings

The automated estimated-age >=18 filter retains visibly child-appearing images and artificial faces. Some aligned crops also have strong rotation, black borders, blur, or occlusion. Most reviewed retained crops preserve a recognizable face, but successful detection and tensor checks do not establish eligibility.

| Selection group | Reviewed | Visual categories |
|---|---:|---|
| random_retained | 50 | {"appears_adult": 36, "age_uncertain": 8, "artificial_or_nonphotographic_face": 1, "appears_child": 5} |
| borderline_retained | 94 | {"appears_adult": 13, "artificial_or_nonphotographic_face": 5, "age_uncertain": 40, "appears_child": 36} |
| under18_sample | 20 | {"appears_child": 16, "artificial_or_nonphotographic_face": 1, "age_uncertain": 2, "appears_adult": 1} |
| detection_failure | 14 | {"age_not_assessable": 6, "not_a_human_face": 3, "age_uncertain": 1, "artificial_or_nonphotographic_face": 3, "appears_adult": 1} |

These are reviewer judgments, not measured age-classification error rates. The combined sample oversamples borderline estimates and failures, so its overall proportions must not be generalized to the full dataset. Even the random subset measures apparent eligibility, not verified adulthood.

## Examples retained by the age model

| Review ID | Filename | Model estimated age | Visual flag |
|---|---|---:|---|
| V022 | Helen_582.png | 22.0 | artificial_or_nonphotographic_face |
| V032 | Lisa_190.png | 24.0 | appears_child |
| V035 | Gabriel_248.png | 24.0 | appears_child |
| V039 | Jonathan_23.png | 73.0 | appears_child |
| V045 | Emily_374.png | 47.0 | appears_child |
| V046 | Abby_534.png | 25.0 | appears_child |
| V054 | Angela_396.png | 22.0 | appears_child |
| V058 | Chris_173.png | 21.0 | appears_child |
| V059 | Christina_138.png | 21.0 | appears_child |
| V060 | Christina_723.png | 21.0 | artificial_or_nonphotographic_face |
| V061 | Christine_779.png | 21.0 | artificial_or_nonphotographic_face |
| V062 | Danielle_213.png | 22.0 | appears_child |

## Next steps

1. Have a teammate independently check the flagged images using the local sheets and `visual_review.csv`; record their judgments in `second_reviewer_decision`.
2. Resolve how child-appearing, artificial, and uncertain faces will be screened before claiming an adult-only dataset. Merely raising the age threshold is not a verified solution.
3. If the full automated pass is run to generate estimates, describe it as provisional screening, preserve all originals and exclusion reasons, and do not finalize class removal or splits until review rules are agreed.
4. Document changes from MI2, including SCRFD instead of RetinaFace and the 160x160 detector input. Final model crops remain 112x112 RGB with CHW float32 values in [-1,1].

## Privacy and files

Keep contact sheets local; they reproduce dataset photographs. The included `.gitignore` excludes the sheets. Do not publish them without verifying applicable rights. The review CSV records filenames and judgments for traceability; individual judgments are provisional.
