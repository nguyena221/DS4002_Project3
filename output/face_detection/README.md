> Current active dataset after exact deduplication: 64,398 images across 100 names; train 45,016, validation 9,665, test 9,717. Age >=25, confidence >=0.5. The Mike/Steve pair and 139 extra exact copies are excluded. See output/final_age25/HANDOFF.md; historical stage counts below are preserved.

> This folder records a completed historical stage or experiment. The active training data uses estimated age >=25 and confidence >=0.5: 64,539 images across 100 names. See output/final_age25/HANDOFF.md. Code was consolidated into four scripts; the commands below use the new names. Original measurements are unchanged.

# Step 1: face detection only

Processed all 80,000 original images. Actual RetinaFace (UniFace ONNX MobileNetV2 implementation) localized one face in 79,505. After preserving 67 prior user-confirmed exclusions, 79,442 remain provisionally usable for the next step.

No age model was loaded. No age cutoff, threshold comparison, class filtering, normalization, or train/validation/test split was applied. Original photos are unchanged. The manifest references the source archive; it does not duplicate the images.

Implementation note: MI2 names InsightFace (RetinaFace). This stage uses actual RetinaFace through UniFace, rather than buffalo_l's SCRFD. Record this library implementation difference. Fixed settings: confidence 0.5, NMS 0.4, input 160x160 near the already cropped images' native size. No threshold experiment was performed. The initial slow 640-input attempt was interrupted and archived, not used in these final results.

Detection confidence and valid landmarks are proxies for usable face localization; this is not a comprehensive image-quality or real-person check. Children, dolls, and illustrations can still be detected. Adult eligibility remains unassessed at this stage.

`face_detection_manifest.csv` separates detector outcomes from prior user exclusions. `name_counts.csv` gives per-name counts; `summary.json` records all settings, counts, versions, and model hash. Stop here for user confirmation before age estimation.
