"""Compare saved thresholds without changing active data.
Age, detection and combined modes reproduce the historical baseline experiments.
Retention counts are not verified child/cartoon classification accuracy."""
import csv, hashlib, json
from collections import Counter
from pathlib import Path
import csv, json, hashlib

def age_read_csv(path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def age_write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def age_main(root, output):
    """compare_updated_age_cutoffs.py: preserved age stage implementation."""
    root = Path(root)
    out = Path(output)
    if out.exists():
        raise RuntimeError('Output exists; preserving prior analysis.')
    prepared_path = root / 'output/prepared_dataset/split_manifest.csv'
    age_path = root / 'output/age_filter/age_filter_manifest.csv'
    audit_path = root / 'data/archive/earlier_experiments/screening_20000/visual_review/visual_review.csv'
    source_hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [prepared_path, age_path]}
    prepared = age_read_csv(prepared_path)
    ages = age_read_csv(age_path)
    audit = age_read_csv(audit_path)
    assert len(prepared) == 73979 and len(ages) == 79442
    by_name = {r['filename']: r for r in prepared}
    age_by_name = {r['filename']: r for r in ages}
    assert len(by_name) == len(prepared)
    for r in prepared:
        source = age_by_name[r['filename']]
        assert r['source_sha256'] == source['sha256'] and r['estimated_age'] == source['estimated_age']
        assert int(r['estimated_age']) >= 18
    development = []
    for r in audit:
        current = by_name.get(r['filename'])
        if current is None or current['split'] != 'train':
            continue
        assert r['sha256'] == current['source_sha256'], 'Visual-audit source hash mismatch'
        development.append(dict(**current, visual_category=r['visual_category'], questionable=r['questionable'], old_review_id=r['review_id']))
    assert development, 'No training-only visual review labels available.'
    thresholds = [18, 25, 30]
    all_names = sorted({r['first_name'] for r in prepared})
    baseline_counts = Counter((r['first_name'] for r in prepared))
    summaries = []
    name_rows = []
    eligible_by_cutoff = {}
    counts_by_cutoff = {}
    for cutoff in thresholds:
        kept = [r for r in prepared if int(r['estimated_age']) >= cutoff]
        counts = Counter((r['first_name'] for r in kept))
        counts_by_cutoff[cutoff] = counts
        eligible = {name for name in all_names if counts[name] >= 300}
        eligible_by_cutoff[cutoff] = eligible
        selected_review = [r for r in development if int(r['estimated_age']) >= cutoff]
        adult_total = sum((r['visual_category'] == 'appears_adult' for r in development))
        adult_kept = sum((r['visual_category'] == 'appears_adult' for r in selected_review))
        flags = sum((r['questionable'] == 'yes' for r in selected_review))
        categories = Counter((r['visual_category'] for r in selected_review))
        summaries.append(dict(estimated_age_cutoff=cutoff, prepared_input_images=len(prepared), retained_before_class_minimum=len(kept), removed_from_current_dataset=len(prepared) - len(kept), retained_percent_of_current_dataset=round(100 * len(kept) / len(prepared), 2), names_meeting_300_minimum=len(eligible), names_below_300_minimum=len(all_names) - len(eligible), images_after_class_minimum=sum((counts[name] for name in eligible)), smallest_class_before_minimum=min((counts[name] for name in all_names)), largest_class_before_minimum=max(counts.values()), training_audit_available=len(development), training_audit_retained=len(selected_review), training_audit_questionable_retained=flags, training_audit_questionable_percent=round(100 * flags / len(selected_review), 2) if selected_review else None, training_audit_child_appearing_retained=categories['appears_child'], training_audit_artificial_or_stylized_retained=categories['artificial_or_strongly_stylized'], training_audit_age_uncertain_retained=categories['age_uncertain'], training_audit_unassessable_retained=categories['not_reliably_assessable'], training_audit_adult_appearing_retained=adult_kept, training_audit_adult_appearing_available=adult_total, training_audit_adult_appearing_removed=adult_total - adult_kept))
        for name in all_names:
            name_rows.append(dict(age_cutoff=cutoff, first_name=name, baseline_images=baseline_counts[name], retained_images=counts[name], meets_300_minimum=name in eligible, images_after_class_minimum=counts[name] if name in eligible else 0))
    out.mkdir()
    age_write_csv(out / 'age_cutoff_comparison.csv', summaries, list(summaries[0]))
    age_write_csv(out / 'name_counts_by_cutoff.csv', name_rows, list(name_rows[0]))
    membership = []
    for r in prepared:
        record = {key: r[key] for key in ['h5_index', 'filename', 'archive_member', 'first_name', 'label', 'split', 'source_sha256', 'estimated_age']}
        for cutoff in thresholds:
            passes = int(r['estimated_age']) >= cutoff
            record[f'age_at_least_{cutoff}'] = passes
            record[f'eligible_after_age_{cutoff}_and_class_minimum'] = passes and r['first_name'] in eligible_by_cutoff[cutoff]
        membership.append(record)
    age_write_csv(out / 'experimental_membership.csv', membership, list(membership[0]))
    review_rows = []
    for r in development:
        record = {key: r[key] for key in ['filename', 'first_name', 'estimated_age', 'visual_category', 'questionable', 'old_review_id']}
        record['current_split'] = 'train'
        for cutoff in thresholds:
            record[f'kept_at_{cutoff}'] = int(r['estimated_age']) >= cutoff
        review_rows.append(record)
    age_write_csv(out / 'training_visual_audit_comparison.csv', review_rows, list(review_rows[0]))
    summary = dict(complete=True, experiment='Age cutoffs only on updated RetinaFace pipeline', cutoffs=thresholds, prepared_images=73979, upstream_age_estimated_images=79442, excluded_under_18_upstream=5463, results=summaries, training_visual_audit_images=len(development), visual_label_source=str(audit_path.relative_to(root)), visual_review_type='Previous assistant judgments of unchanged original images, matched by filename and source hash', current_test_images_used_for_quality_selection=False, new_visual_validation_performed=False, actual_age_labels_available=False, detection_settings_changed=False, model_rerun=False, prepared_dataset_modified=False, splits_modified=False, source_hashes=source_hashes, warning='Counts are exact for saved model estimates; visual diagnostics are small-sample assistant judgments, not verified age accuracy or a fresh validation.')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    table = '\n'.join((f"| {r['estimated_age_cutoff']} | {r['retained_before_class_minimum']:,} | {r['removed_from_current_dataset']:,} | {r['names_meeting_300_minimum']} | {r['images_after_class_minimum']:,} |" for r in summaries))
    quality = '\n'.join((f"| {r['estimated_age_cutoff']} | {r['training_audit_questionable_retained']}/{r['training_audit_retained']} | {r['training_audit_questionable_percent']}% | {r['training_audit_child_appearing_retained']} | {r['training_audit_artificial_or_stylized_retained']} | {r['training_audit_adult_appearing_retained']}/{r['training_audit_adult_appearing_available']} |" for r in summaries))
    report = f'# Updated dataset: age-cutoff comparison only\n\nCompared the agreed estimated-age cutoffs 18, 25, and 30 using the saved buffalo_l age estimates derived from the actual RetinaFace bounding boxes. No detector settings or model weights were changed and no model inference was rerun. Current HDF5 data, split manifests, and filtering decisions remain unchanged.\n\n## Exact retention counts\n\nThe comparison starts with the current 73,979 prepared images, already estimated >=18. The upstream 5,463 estimated-under-18 images remain excluded and are not reintroduced.\n\n| Estimated-age cutoff | Images retained | Additional images removed | Names meeting >=300 | Images after >=300 rule |\n|---:|---:|---:|---:|---:|\n{table}\n\nClass-minimum effects are hypothetical, reported separately for each cutoff. No new name categories have actually been removed.\n\n## Existing visual quality check: training images only\n\nReused {len(development)} earlier assistant judgments that match current training images by filename and original hash. Judgments describe original image content, so remain applicable even though the detector/age estimates changed. Current RetinaFace-derived age estimates were used for this comparison, not the old SCRFD ages. Validation/test images were excluded from these quality diagnostics. This is a development check, not a fresh independent validation or verified age ground truth.\n\n| Cutoff | Questionable / retained reviewed | Flag rate | Child-appearing kept | Artificial/stylized kept | Adult-appearing kept |\n|---:|---:|---:|---:|---:|---:|\n{quality}\n\nQuestionable includes age uncertainty and unassessable images; it does not mean every flagged image is definitely ineligible. Higher cutoffs may remove true adults, especially younger adults. A smaller retained set is not automatically a more accurate one. Do not reuse old SCRFD-based threshold percentages as current-pipeline results.\n\n## What remains unchanged\n\n- Current prepared dataset: 73,979 images, estimated-age cutoff 18.\n- Existing train/validation/test assignments and label mapping.\n- Original source photographs and confirmed manual exclusions.\n- RetinaFace confidence and other detection settings.\n\nNo new cutoff has been adopted. Before adopting one, discuss the data/quality tradeoff and arrange fresh validation. A changed cutoff requires updating the active manifests and checking the split/class distributions; this experiment does not silently replace them. Detection-threshold testing has not been performed in this step.\n\n## Files\n\n- age_cutoff_comparison.csv: exact counts and training-only visual diagnostics.\n- name_counts_by_cutoff.csv: all 100 name counts for every cutoff.\n- experimental_membership.csv: flags mapped to current HDF5 indices and current splits; these are experimental filters, not new splits.\n- training_visual_audit_comparison.csv: image-level development judgments and updated threshold outcomes.\n- summary.json: provenance, assumptions, and source-file hashes.\n'
    (out / 'README.md').write_text(report, encoding='utf-8')
    for path, expected in source_hashes.items():
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == expected
    assert summaries[0]['retained_before_class_minimum'] == 73979
    assert all((summaries[i]['retained_before_class_minimum'] >= summaries[i + 1]['retained_before_class_minimum'] for i in range(2)))
    print(json.dumps(summary, indent=2))

def detection_read(p):
    with p.open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))

def detection_write(p, rows):
    with p.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def detection_main(root, output):
    """compare_updated_detection_thresholds.py: preserved detection stage implementation."""
    root = Path(root)
    out = Path(output)
    if out.exists():
        raise RuntimeError('Preserving existing comparison; output already exists')
    paths = [root / 'output/prepared_dataset/split_manifest.csv', root / 'output/age_filter/age_filter_manifest.csv', root / 'data/archive/earlier_experiments/screening_20000/visual_review/visual_review.csv']
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    prepared, ages, audit = map(detection_read, paths)
    lookup = {r['filename']: r for r in ages}
    current = {r['filename']: r for r in prepared}
    assert len(current) == len(prepared)
    for r in prepared:
        a = lookup[r['filename']]
        assert a['sha256'] == r['source_sha256'] and int(a['estimated_age']) >= 18
        r['retinaface_score'] = float(a['retinaface_score'])
        assert r['retinaface_score'] >= 0.5
    review = []
    for a in audit:
        r = current.get(a['filename'])
        if r is not None and r['split'] == 'train':
            assert r['source_sha256'] == a['sha256']
            review.append(dict(r, visual_category=a['visual_category'], questionable=a['questionable']))
    names = sorted({r['first_name'] for r in prepared})
    results = []
    name_rows = []
    membership = []
    review_rows = []
    thresholds = [0.5, 0.7, 0.9]
    eligible = {}
    for t in thresholds:
        kept = [r for r in prepared if r['retinaface_score'] >= t]
        counts = Counter((r['first_name'] for r in kept))
        eligible[t] = {n for n in names if counts[n] >= 300}
        reviewed = [r for r in review if r['retinaface_score'] >= t]
        cats = Counter((r['visual_category'] for r in reviewed))
        flags = sum((r['questionable'] == 'yes' for r in reviewed))
        results.append(dict(detection_threshold=t, estimated_age_cutoff=18, images_retained=len(kept), additional_images_removed=len(prepared) - len(kept), names_meeting_300_minimum=len(eligible[t]), images_after_class_minimum=sum((counts[n] for n in eligible[t])), training_audit_retained=len(reviewed), training_audit_questionable_retained=flags, training_audit_questionable_percent=round(100 * flags / len(reviewed), 2) if reviewed else None, child_appearing_retained=cats['appears_child'], artificial_or_stylized_retained=cats['artificial_or_strongly_stylized'], adult_appearing_retained=cats['appears_adult'], adult_appearing_removed=sum((r['visual_category'] == 'appears_adult' for r in review)) - cats['appears_adult']))
        name_rows.extend((dict(detection_threshold=t, first_name=n, images_retained=counts[n], meets_300_minimum=n in eligible[t]) for n in names))
    for r in prepared:
        item = {k: r[k] for k in ['h5_index', 'filename', 'first_name', 'split', 'source_sha256', 'estimated_age', 'retinaface_score']}
        for t in thresholds:
            item[f'kept_at_{t}'] = r['retinaface_score'] >= t
            item[f'eligible_after_{t}_and_class_minimum'] = r['retinaface_score'] >= t and r['first_name'] in eligible[t]
        membership.append(item)
    for r in review:
        item = {k: r[k] for k in ['filename', 'first_name', 'retinaface_score', 'visual_category', 'questionable']}
        for t in thresholds:
            item[f'kept_at_{t}'] = r['retinaface_score'] >= t
        review_rows.append(item)
    assert results[0]['images_retained'] == len(prepared)
    assert all((results[i]['images_retained'] >= results[i + 1]['images_retained'] for i in range(2)))
    for p in paths:
        assert hashlib.sha256(p.read_bytes()).hexdigest() == hashes[str(p.relative_to(root))]
    out.mkdir()
    detection_write(out / 'detection_threshold_comparison.csv', results)
    detection_write(out / 'name_counts_by_threshold.csv', name_rows)
    detection_write(out / 'experimental_membership.csv', membership)
    detection_write(out / 'training_visual_audit_comparison.csv', review_rows)
    summary = dict(complete=True, experiment='Saved RetinaFace confidence filtering on current prepared dataset', thresholds=thresholds, estimated_age_cutoff=18, prepared_images=len(prepared), training_visual_audit_images=len(review), results=results, source_hashes=hashes, model_rerun=False, prepared_dataset_modified=False, splits_modified=False, new_visual_validation_performed=False, verified_age_or_detection_ground_truth_available=False, current_test_images_used_for_quality_selection=False, limitation='Post-filter comparison of existing single-face prepared images. Not a full raw-data detector rerun: excluded multiple-face images are not reconsidered. Confidence is not an age or photographic-realism score. Visual diagnostics reuse small-sample assistant judgments, not ground truth.')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    table = '\n'.join((f"| {r['detection_threshold']} | {r['images_retained']:,} | {r['additional_images_removed']:,} | {r['names_meeting_300_minimum']} | {r['training_audit_questionable_retained']}/{r['training_audit_retained']} ({r['training_audit_questionable_percent']}%) |" for r in results))
    (out / 'README.md').write_text('# Detection confidence comparison on updated dataset\n\nCompares saved RetinaFace confidence scores at 0.5, 0.7, and 0.9. Age cutoff stays at 18; original image preparation, manual exclusions, and train/validation/test assignments remain unchanged. Reproduce with `python scripts/compare_thresholds.py detection` from a checkout without this output folder. Existing output is protected against overwrite.\n\n| Confidence cutoff | Images retained | Additional removed | Names meeting 300 minimum | Questionable among reviewed retained |\n|---:|---:|---:|---:|---:|\n' + table + '\n\nCounts apply to the current prepared dataset, not a new full 80,000-image detection run. This post-filter comparison does not reconsider previously excluded multi-face images that could become single-face at another cutoff. Model weights, input size and NMS are fixed. No model inference was rerun.\n\nVisual diagnostics reuse earlier assistant judgments matched by original file hash and restricted to the current training split. These are neither verified age labels nor a fresh independent review. Questionable includes uncertain ages and unassessable images. Percentages are descriptive of this small reviewed subset and do not estimate whole-dataset error rates. Confidence measures detector confidence, not adulthood, realism, or correct identity.\n\nNo threshold has been adopted. Class-minimum effects are hypothetical. CSV files record summary results, per-name counts, image-level experimental membership, and training-only visual judgments. summary.json records source hashes and limitations. Assertions check baseline retention, nested thresholds, matching source hashes, and unchanged input manifests. No detection precision/recall can be established without annotated ground truth.\n', encoding='utf-8')
    print(json.dumps(results, indent=2))

def combined_read(p):
    with p.open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))

def combined_main(root, output):
    """compare_age_detection.py: preserved combined stage implementation."""
    root = Path(root)
    out = Path(output)
    if out.exists():
        raise RuntimeError('Output exists; preserving earlier results')
    paths = [root / 'output/prepared_dataset/split_manifest.csv', root / 'output/age_filter/age_filter_manifest.csv']
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    prepared, ages = map(combined_read, paths)
    lookup = {r['filename']: r for r in ages}
    names = sorted({r['first_name'] for r in prepared})
    for r in prepared:
        a = lookup[r['filename']]
        assert r['source_sha256'] == a['sha256'] and r['estimated_age'] == a['estimated_age']
        r['score'] = float(a['retinaface_score'])
        assert r['score'] >= 0.5
    results = []
    for age in [18, 25, 30]:
        baseline = [r for r in prepared if int(r['estimated_age']) >= age]
        for det in [0.5, 0.7, 0.9]:
            kept = [r for r in baseline if r['score'] >= det]
            counts = Counter((r['first_name'] for r in kept))
            eligible = {n for n in names if counts[n] >= 300}
            results.append(dict(age_cutoff=age, detection_threshold=det, images_before_class_minimum=len(kept), additional_removed_vs_same_age_det_05=len(baseline) - len(kept), names_meeting_300_minimum=len(eligible), images_after_class_minimum=sum((counts[n] for n in eligible)), names_below_minimum=';'.join((n for n in names if n not in eligible))))
    assert results[0]['images_before_class_minimum'] == len(prepared)
    for i in [0, 3, 6]:
        assert results[i]['images_before_class_minimum'] >= results[i + 1]['images_before_class_minimum'] >= results[i + 2]['images_before_class_minimum']
    for p in paths:
        assert hashlib.sha256(p.read_bytes()).hexdigest() == hashes[str(p.relative_to(root))]
    out.mkdir()
    with (out / 'comparison.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    summary = dict(complete=True, results=results, source_hashes=hashes, model_rerun=False, active_dataset_modified=False, splits_modified=False, warning='Post-filter comparison on existing single-face prepared images. Does not reconsider previously excluded multiple-face images. Exact retention counts are not child/cartoon classification or detection accuracy.')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    table = '\n'.join((f"| {r['age_cutoff']} | {r['detection_threshold']} | {r['images_before_class_minimum']:,} | {r['additional_removed_vs_same_age_det_05']} | {r['images_after_class_minimum']:,} | {r['names_meeting_300_minimum']} |" for r in results))
    (out / 'README.md').write_text('# Age and detection confidence comparison\n\nAll nine combinations of saved estimated-age cutoffs 18/25/30 and RetinaFace confidence cutoffs 0.5/0.7/0.9 on the current prepared dataset. Reproduce with `python scripts/compare_thresholds.py combined` when this output folder does not already exist. Active data, manual exclusions, labels and splits are unchanged.\n\n| Age | Detection | Kept before class minimum | Extra removed versus detection 0.5 at same age | Kept after class minimum | Names retained |\n|---:|---:|---:|---:|---:|---:|\n' + table + '\n\nThe class minimum is 300 images per name, calculated separately for every combination. These are hypothetical filters, not newly adopted settings. Saved model estimates are reused; detector inference is not rerun. Previously excluded multi-face images are not reconsidered, so this is not a full raw-data detector rerun.\n\nThese counts cannot tell how many children or artificial characters remain, nor whether removed images were unwanted. No whole-dataset visual labels are available. Higher confidence is not evidence of adulthood or photographic realism. No new visual validation or accuracy measurement was performed. Source hashes and assertions verify matching input records, unchanged manifests and nested retention counts.\n', encoding='utf-8')
    print(json.dumps(results, indent=2))


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['age', 'detection', 'combined', 'all'])
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1], help='Repository containing the preserved baseline manifests')
    parser.add_argument('--output', type=Path, help='New output directory; for all, a parent containing three result folders')
    args = parser.parse_args(argv)
    folders = {'age': 'age_cutoff_comparison', 'detection': 'detection_threshold_comparison', 'combined': 'age_detection_comparison'}
    handlers = {'age': age_main, 'detection': detection_main, 'combined': combined_main}
    modes = list(handlers) if args.mode == 'all' else [args.mode]
    outputs = {mode: (args.output / folders[mode] if args.mode == 'all' else args.output) if args.output else args.root / 'output' / folders[mode] for mode in modes}
    for path in outputs.values():
        if path.exists():
            parser.error(f'Output exists; preserving previous results: {path}. Choose a new --output path.')
    for mode, path in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers[mode](args.root, path)

if __name__ == '__main__':
    main()
