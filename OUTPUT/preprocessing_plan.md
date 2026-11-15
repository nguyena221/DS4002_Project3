# Preprocessing implementation plan

Owner: Annie Nguyen. Status: not implemented; original images preserved.

## MI2 sequence

1. Inventory filenames and labels; flag unreadable or invalid files.
2. Detect faces; flag images without a clear single detected face.
3. Estimate age using InsightFace `buffalo_l`; retain estimated age >=18.
4. Remove name categories with fewer than 300 eligible images.
5. Prepare aligned RGB crops of 112 x 112 pixels and normalize to [-1, 1].
6. Produce name-stratified 70% train, 15% validation, and 15% test splits.
7. Hand off processed inputs, label mapping, split manifests, and screening counts.

## Decisions to resolve before the full run

- Pilot first: proposed 10 randomly selected images per name, 1,000 total.
  This pilot is an implementation check, not an additional MI2 requirement.
- MI2 says RetinaFace; the documented legacy `buffalo_l` pack uses SCRFD.
  Record the actual detector and any deviation rather than calling it RetinaFace.
- Pin InsightFace/model versions, hardware/provider, detector settings, and random seed.
- Estimated age >=18 does not verify adulthood. Inspect pilot age estimates,
  especially borderline examples, and document remaining uncertainty.
- A 300-image minimum does not eliminate class imbalance. Report retained counts.
- Check exact duplicate images before splitting; do not claim verified identity
  separation without supporting identifiers.
- Agree with Eli on alignment, channel order, tensor layout (C,H,W), and the
  exact normalization formula so normalization is not applied twice.
- Keep rejection reasons and intermediate estimates in a resumable manifest.
- Record changes from MI2 and their reasons. Do not adjust screening based on
  held-out model performance.

## Existing completed work

Full raw-archive EDA decoded 80,000 images successfully and confirmed 800 images
for each of 100 names. This does not validate age, labels, or unique identities.
