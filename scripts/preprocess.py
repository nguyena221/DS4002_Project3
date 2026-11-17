"""Preprocess Name100 using separately runnable stages.
Final settings: age25, confidence0.5, minimum300; tensors and prior splits preserved.
Baseline stages intentionally reproduce age18 before the final approved update."""
import argparse, csv, hashlib, importlib.metadata, io, json, os, tarfile, time
from collections import Counter
from pathlib import Path
import cv2, numpy as np, onnxruntime as ort
from PIL import Image, UnidentifiedImageError
import argparse, csv, hashlib, importlib.metadata, json, tarfile, time
from insightface.app.common import Face
from insightface.model_zoo.attribute import Attribute
import argparse, csv, hashlib, json
import argparse, csv, hashlib, importlib.metadata, json, random, tarfile, time
from collections import Counter, defaultdict
import cv2, h5py, numpy as np
from PIL import Image, ImageDraw, ImageFont
from insightface.utils import face_align
from sklearn.model_selection import train_test_split
import csv, hashlib, json, time
import h5py, numpy as np
from prepared_dataset import PreparedFaces

def detect_main(argv=None):
    """face_detection.py: preserved detect stage implementation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('output/face_detection'))
    args = parser.parse_args(argv)
    root = Path.cwd()
    started = time.time()
    if args.output.exists():
        parser.error('Output exists; preserve it and choose a new path.')
    if not args.archive.is_file():
        parser.error('Dataset archive is missing.')
    os.environ['UNIFACE_CACHE_DIR'] = str((root / 'models/retinaface').resolve())
    from uniface.detection import RetinaFace
    from uniface.constants import RetinaFaceWeights
    print('Loading actual RetinaFace detector (no age model)...', flush=True)
    detector = RetinaFace(model_name=RetinaFaceWeights.MNET_V2, input_size=(160, 160), providers=['CPUExecutionProvider'])
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    detector.session = ort.InferenceSession(detector._model_path, sess_options=opts, providers=['CPUExecutionProvider'])
    known = {}
    decisions = root / 'data/user_review_exclusions.csv'
    if decisions.is_file():
        with decisions.open(newline='', encoding='utf-8') as f:
            known = {r['filename']: r for r in csv.DictReader(f) if r['decision'] == 'exclude'}
    args.output.mkdir(parents=True)
    fields = ['archive_member', 'filename', 'first_name', 'sha256', 'width', 'height', 'face_count', 'top_detection_score', 'detections_json', 'face_detection_status', 'face_detection_reason', 'prior_user_excluded', 'prior_review_id', 'status', 'reason']
    face_outcomes = Counter()
    outcomes = Counter()
    raw_counts = Counter()
    single_counts = Counter()
    retained_counts = Counter()
    errors = []
    done = 0
    seen = set()
    n_prior = 0
    with (args.output / 'face_detection_manifest.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        with tarfile.open(args.archive, 'r|gz') as archive:
            for member in archive:
                if not member.isfile() or not member.name.lower().endswith('.png'):
                    continue
                filename = member.name.rsplit('/', 1)[-1]
                data = archive.extractfile(member).read()
                digest = hashlib.sha256(data).hexdigest()
                if filename in seen:
                    raise ValueError(f'Duplicate filename: {filename}')
                seen.add(filename)
                row = dict.fromkeys(fields, '')
                row.update(archive_member=member.name, filename=filename, sha256=digest, face_detection_status='excluded', prior_user_excluded='no')
                try:
                    name, number = Path(filename).stem.rsplit('_', 1)
                except ValueError:
                    name = ''
                    number = ''
                row['first_name'] = name
                raw_counts[name] += 1
                if not name.isalpha() or not number.isdigit():
                    reason = 'invalid_name_label'
                else:
                    try:
                        with Image.open(io.BytesIO(data)) as image:
                            image.load()
                            rgb = np.asarray(image.convert('RGB'))
                            row['width'], row['height'] = image.size
                        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    except (UnidentifiedImageError, OSError, ValueError) as error:
                        bgr = None
                        errors.append(dict(filename=filename, error=str(error)))
                    if bgr is None:
                        reason = 'corrupted_or_unreadable_image'
                    else:
                        faces = detector.detect(bgr)
                        row['face_count'] = len(faces)
                        details = [dict(bbox=np.asarray(face.bbox).tolist(), score=float(face.confidence), landmarks=np.asarray(face.landmarks).tolist() if face.landmarks is not None else None) for face in faces]
                        row['detections_json'] = json.dumps(details, separators=(',', ':'))
                        if faces:
                            row['top_detection_score'] = max((float(face.confidence) for face in faces))
                        if not faces:
                            reason = 'no_face_detected'
                        elif len(faces) > 1:
                            reason = 'multiple_faces_detected'
                        else:
                            face = faces[0]
                            box = np.asarray(face.bbox)
                            lm = np.asarray(face.landmarks) if face.landmarks is not None else None
                            if box.shape != (4,) or not np.isfinite(box).all() or box[2] <= box[0] or (box[3] <= box[1]):
                                reason = 'invalid_face_box'
                            elif lm is None or lm.shape != (5, 2) or (not np.isfinite(lm).all()):
                                reason = 'invalid_face_landmarks'
                            else:
                                reason = 'single_face_localized'
                                row['face_detection_status'] = 'retained'
                                single_counts[name] += 1
                row['face_detection_reason'] = reason
                row['status'] = row['face_detection_status']
                row['reason'] = reason
                if filename in known:
                    assert digest == known[filename]['sha256'], 'Known exclusion hash mismatch'
                    row['prior_user_excluded'] = 'yes'
                    row['prior_review_id'] = known[filename]['review_id']
                    row['status'] = 'excluded'
                    row['reason'] = 'prior_user_confirmed_exclusion'
                    n_prior += 1
                if row['status'] == 'retained':
                    retained_counts[name] += 1
                face_outcomes[row['face_detection_reason']] += 1
                outcomes[row['reason']] += 1
                writer.writerow(row)
                done += 1
                if done % 1000 == 0:
                    f.flush()
                    print(f'Face detection {done:,}/80,000; single-face images {sum(single_counts.values()):,}; elapsed {time.time() - started:.0f}s', flush=True)
    assert done == 80000 and len(raw_counts) == 100 and (set(raw_counts.values()) == {800})
    assert n_prior == len(known)
    with (args.output / 'name_counts.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['first_name', 'raw_images', 'single_face_localized', 'retained_after_prior_user_exclusions'])
        w.writerows(((name, raw_counts[name], single_counts[name], retained_counts[name]) for name in sorted(raw_counts)))
    summary = dict(stage='1: readability and face detection only', complete=True, raw_images=done, raw_names=len(raw_counts), detector='RetinaFace', implementation='UniFace', model='MNET_V2', model_path=str(Path(detector._model_path).resolve()), model_sha256=hashlib.sha256(Path(detector._model_path).read_bytes()).hexdigest(), source_code='https://github.com/yakhyo/uniface', retinaface_reference='https://github.com/deepinsight/insightface/tree/master/detection/retinaface', confidence_threshold=detector.confidence_threshold, nms_threshold=detector.nms_threshold, input_size=list(detector.input_size), face_detection_outcomes=dict(face_outcomes), retained_single_face=sum(single_counts.values()), prior_user_exclusions_preserved=n_prior, retained_after_prior_user_exclusions=sum(retained_counts.values()), final_stage_outcomes=dict(outcomes), corrupt_image_errors=errors, age_model_loaded=False, age_filter_applied=False, threshold_comparisons=False, class_minimum_applied=False, model_input_tensors_created=False, splits_created=False, original_images_modified=False, versions={p: importlib.metadata.version(p) for p in ['uniface', 'onnxruntime', 'numpy', 'Pillow', 'opencv-python-headless']}, elapsed_seconds=round(time.time() - started, 1))
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    (args.output / 'README.md').write_text(f"# Step 1: face detection only\n\nProcessed all {done:,} original images. Actual RetinaFace (UniFace ONNX MobileNetV2 implementation) localized one face in {sum(single_counts.values()):,}. After preserving {n_prior} prior user-confirmed exclusions, {sum(retained_counts.values()):,} remain provisionally usable for the next step.\n\nNo age model was loaded. No age cutoff, threshold comparison, class filtering, normalization, or train/validation/test split was applied. Original photos are unchanged. The manifest references the source archive; it does not duplicate the images.\n\nImplementation note: MI2 names InsightFace (RetinaFace). This stage uses actual RetinaFace through UniFace, rather than buffalo_l's SCRFD. Record this library implementation difference. Fixed settings: confidence 0.5, NMS 0.4, input 160x160 near the already cropped images' native size. No threshold experiment was performed. The initial slow 640-input attempt was interrupted and archived, not used in these final results.\n\nDetection confidence and valid landmarks are proxies for usable face localization; this is not a comprehensive image-quality or real-person check. Children, dolls, and illustrations can still be detected. Adult eligibility remains unassessed at this stage.\n\n`face_detection_manifest.csv` separates detector outcomes from prior user exclusions. `name_counts.csv` gives per-name counts; `summary.json` records all settings, counts, versions, and model hash. Stop here for user confirmation before age estimation.\n", encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)

def estimate_age_main(argv=None):
    """age_filter.py: preserved estimate_age stage implementation."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--face-manifest', type=Path, default=Path('output/face_detection/face_detection_manifest.csv'))
    p.add_argument('--output', type=Path, default=Path('output/age_filter'))
    p.add_argument('--model', type=Path, default=Path('models/models/buffalo_l/genderage.onnx'))
    args = p.parse_args(argv)
    started = time.time()
    cutoff = 18
    if args.output.exists():
        p.error('Output already exists. Preserve it and choose a new output path.')
    if not args.archive.is_file() or not args.model.is_file():
        p.error('Original archive or buffalo_l genderage.onnx is missing.')
    face_summary = json.loads((args.face_manifest.parent / 'summary.json').read_text())
    assert face_summary['complete'] and face_summary['raw_images'] == 80000
    with args.face_manifest.open(newline='', encoding='utf-8') as f:
        all_faces = list(csv.DictReader(f))
    selected = {r['archive_member']: r for r in all_faces if r['status'] == 'retained'}
    assert len(selected) == 79442 == face_summary['retained_after_prior_user_exclusions']
    assert all((r['face_count'] == '1' and r['prior_user_excluded'] == 'no' for r in selected.values()))
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.model.resolve()), sess_options=opts, providers=['CPUExecutionProvider'])
    model = Attribute(model_file=str(args.model.resolve()), session=session)
    assert model.taskname == 'genderage'
    args.output.mkdir(parents=True)
    fields = ['archive_member', 'filename', 'first_name', 'sha256', 'retinaface_score', 'retinaface_bbox_json', 'estimated_age', 'age_cutoff', 'status', 'reason']
    outcomes = Counter()
    name_input = Counter()
    name_kept = Counter()
    age_counts = Counter()
    seen = set()
    done = 0
    print(f'Age estimation only: {len(selected):,} retained images; cutoff fixed at 18.', flush=True)
    with (args.output / 'age_filter_manifest.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        with tarfile.open(args.archive, 'r|gz') as archive:
            for member in archive:
                if member.name not in selected:
                    continue
                source = selected[member.name]
                data = archive.extractfile(member).read()
                assert hashlib.sha256(data).hexdigest() == source['sha256'], 'Source image changed since face detection.'
                assert member.name not in seen
                seen.add(member.name)
                image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError(f'Previously decoded source no longer opens: {member.name}')
                detections = json.loads(source['detections_json'])
                assert len(detections) == 1
                box = np.asarray(detections[0]['bbox'], dtype=np.float32)
                assert box.shape == (4,) and np.isfinite(box).all()
                face = Face(bbox=box)
                _, age = model.get(image, face)
                assert isinstance(age, (int, np.integer))
                status = 'retained' if age >= cutoff else 'excluded'
                reason = 'estimated_age_at_least_18' if status == 'retained' else 'estimated_age_under_18'
                row = dict(archive_member=member.name, filename=source['filename'], first_name=source['first_name'], sha256=source['sha256'], retinaface_score=source['top_detection_score'], retinaface_bbox_json=json.dumps(box.tolist(), separators=(',', ':')), estimated_age=int(age), age_cutoff=cutoff, status=status, reason=reason)
                w.writerow(row)
                outcomes[reason] += 1
                name_input[source['first_name']] += 1
                age_counts[int(age)] += 1
                if status == 'retained':
                    name_kept[source['first_name']] += 1
                done += 1
                if done % 1000 == 0:
                    f.flush()
                    print(f"Age estimates {done:,}/{len(selected):,}; retained {outcomes['estimated_age_at_least_18']:,}; elapsed {time.time() - started:.0f}s", flush=True)
    assert done == len(selected) and seen == selected.keys()
    with (args.output / 'name_counts.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['first_name', 'input_after_face_detection', 'estimated_age_at_least_18', 'estimated_age_under_18'])
        w.writerows(((n, name_input[n], name_kept[n], name_input[n] - name_kept[n]) for n in sorted(name_input)))
    with (args.output / 'estimated_age_counts.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['estimated_age', 'images'])
        w.writerows(sorted(age_counts.items()))
    summary = dict(stage='2: estimated age >=18 only', complete=True, input_images=done, retained=outcomes['estimated_age_at_least_18'], excluded_under_18=outcomes['estimated_age_under_18'], age_cutoff=18, outcomes=dict(outcomes), model='InsightFace buffalo_l genderage.onnx', api='InsightFace Attribute.get; same module underlying FaceAnalysis', face_boxes='Existing RetinaFace step-1 bounding boxes; no face redetection', model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(), model_input_size=list(model.input_size), gender_output_saved=False, provider='CPUExecutionProvider', versions={x: importlib.metadata.version(x) for x in ['insightface', 'onnxruntime', 'numpy', 'opencv-python-headless']}, original_dataset_modified=False, step1_outputs_modified=False, prior_user_exclusions_reintroduced=False, age_cutoff_comparisons=False, detection_threshold_comparisons=False, class_minimum_applied=False, saved_training_crops_or_tensors=False, normalization_for_training_applied=False, splits_created=False, elapsed_seconds=round(time.time() - started, 1), warning='Estimated age is not verified age; false inclusion/exclusion remains possible. Retained means model-estimated age >=18, not confirmed adulthood.')
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    (args.output / 'README.md').write_text(f"# Step 2: estimated-age filtering only\n\nProcessed all {done:,} images retained after face detection and prior user exclusions. The buffalo_l age model estimated age >=18 for {summary['retained']:,} and under 18 for {summary['excluded_under_18']:,}. Original images and Step 1 outputs are unchanged.\n\nUses InsightFace's `Attribute.get`, the age module underlying `FaceAnalysis`, with the existing RetinaFace bounding boxes. No detector was rerun. The joint model's gender output is discarded and not saved. The model's internal 96x96 inference crop is temporary and is not the later 112x112 training input.\n\nAge cutoff fixed at 18 as specified in MI2; no age or detector threshold comparisons. No class-size filtering, saved training crops/tensors, normalization for training, or train/validation/test splits. Stop here for user confirmation.\n\nThese are estimated ages, not verified adulthood. The previously observed age-model errors remain a limitation. A changed face detector changes boxes supplied to the age module; do not treat earlier pilot error rates as a direct measurement of this new stage.\n\nFiles: `age_filter_manifest.csv` records every input and age decision; `name_counts.csv` gives per-name counts without dropping any names; `estimated_age_counts.csv` gives aggregate estimated-age frequencies; `summary.json` records settings, counts, versions, and model hash. Images remain in the original archive.\n", encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)

def filter_classes_main(argv=None):
    """class_filter.py: preserved filter_classes stage implementation."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=Path('output/age_filter/age_filter_manifest.csv'))
    p.add_argument('--output', type=Path, default=Path('output/class_filter'))
    args = p.parse_args(argv)
    if args.output.exists():
        p.error('Output exists; preserve it and choose another path.')
    upstream = json.loads((args.input.parent / 'summary.json').read_text())
    assert upstream['complete'] and upstream['age_cutoff'] == 18
    with args.input.open(newline='', encoding='utf-8') as f:
        age_rows = list(csv.DictReader(f))
    retained = [r for r in age_rows if r['status'] == 'retained']
    assert len(retained) == upstream['retained'] == 73979
    assert all((int(r['estimated_age']) >= 18 for r in retained))
    assert len({r['archive_member'] for r in retained}) == len(retained)
    counts = Counter((r['first_name'] for r in retained))
    assert len(counts) == 100
    threshold = 300
    eligible = {n for n, c in counts.items() if c >= threshold}
    excluded = {n: c for n, c in sorted(counts.items()) if c < threshold}
    args.output.mkdir(parents=True)
    fields = list(retained[0]) + ['age_filter_status', 'class_retained_count', 'minimum_class_size']
    final = []
    with (args.output / 'class_filter_manifest.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for source in retained:
            row = dict(source, age_filter_status=source['status'], class_retained_count=counts[source['first_name']], minimum_class_size=threshold)
            row['status'] = 'retained' if source['first_name'] in eligible else 'excluded'
            row['reason'] = 'name_meets_300_minimum' if row['status'] == 'retained' else 'name_fewer_than_300_images'
            w.writerow(row)
            if row['status'] == 'retained':
                final.append(row)
    with (args.output / 'retained_manifest.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(final)
    with (args.output / 'name_counts.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['first_name', 'age_screened_images', 'minimum_class_size', 'class_status', 'images_after_class_filter'])
        w.writerows(((n, counts[n], threshold, 'retained' if n in eligible else 'excluded', counts[n] if n in eligible else 0) for n in sorted(counts)))
    assert sum((counts[n] for n in eligible)) == len(final)
    assert len(final) + sum(excluded.values()) == len(retained)
    source_by_name = {r['archive_member']: r for r in retained}
    for row in final:
        source = source_by_name[row['archive_member']]
        assert row['sha256'] == source['sha256'] and row['first_name'] == source['first_name']
        assert row['estimated_age'] == source['estimated_age'] and counts[row['first_name']] >= threshold
    summary = dict(stage='3: minimum 300 retained images per first name', complete=True, input_images=len(retained), input_names=len(counts), minimum_class_size=threshold, retained_images=len(final), retained_names=len(eligible), excluded_images=sum(excluded.values()), excluded_names=len(excluded), excluded_name_counts=excluded, minimum_input_class_count=min(counts.values()), maximum_input_class_count=max(counts.values()), smallest_classes=[dict(first_name=n, count=c) for n, c in sorted(counts.items(), key=lambda t: (t[1], t[0]))[:5]], source_manifest_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(), original_images_modified=False, age_results_modified=False, age_estimation_rerun=False, resized_training_inputs=False, normalization_applied=False, splits_created=False, verification_passed=True, warning='Eligibility uses model-estimated ages, not verified adulthood. A minimum class size does not guarantee equal class sizes.')
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    (args.output / 'README.md').write_text(f'# Step 3: minimum class size only\n\nApplied the MI2 rule: keep a first name only when it has at least 300 images remaining after the estimated-age >=18 filter. Exactly 300 qualifies.\n\nInput: {len(retained):,} images across {len(counts)} names. Retained: {len(final):,} images across {len(eligible)} names. Excluded: {sum(excluded.values()):,} images across {len(excluded)} names. Smallest input class: {min(counts.values())}; largest: {max(counts.values())}.\n\nNo source photographs or age results were changed. No detector/age model reruns, image resizing, training normalization, or dataset splits were performed. Stop here for user confirmation.\n\n`name_counts.csv` provides every name and its eligibility. `class_filter_manifest.csv` records the decision for each age-retained image. `retained_manifest.csv` is the input list for the next authorized step. `summary.json` records totals, settings, verification, and the source manifest hash.\n\nAdult eligibility is based on estimated age. The 300-image minimum does not make classes equally sized.\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))

def prepare_write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def prepare_make_splits(rows, labels, seed):
    n = len(rows)
    ids = np.arange(n)
    train, hold = train_test_split(ids, train_size=0.7, stratify=labels, random_state=seed)
    validation, test = train_test_split(hold, test_size=0.5, stratify=labels[hold], random_state=seed + 1)
    assignment = np.full(n, -1, dtype=np.int8)
    for split, indices in enumerate([train, validation, test]):
        assignment[indices] = split
    initial = assignment.copy()
    before = Counter(((int(labels[i]), int(initial[i])) for i in ids))
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        a, b = (find(a), find(b))
        if a != b:
            parent[b] = a
    for key in ['sha256', 'prepared_rgb_sha256']:
        first = {}
        for i, row in enumerate(rows):
            value = row[key]
            if value in first:
                union(i, first[value])
            else:
                first[value] = i
    grouped = defaultdict(list)
    for i in ids:
        grouped[find(int(i))].append(int(i))
    duplicates = [g for g in grouped.values() if len(g) > 1]
    duplicate_ids = {i for group in duplicates for i in group}
    unique = np.array([i not in duplicate_ids for i in ids])
    rng = random.Random(seed)
    swaps = 0
    for group in sorted(duplicates, key=lambda g: rows[min(g)]['filename']):
        counts = Counter((int(assignment[i]) for i in group))
        largest = max(counts.values())
        target = rng.choice(sorted((s for s, c in counts.items() if c == largest)))
        for i in group:
            if assignment[i] == target:
                continue
            candidates = np.flatnonzero((labels == labels[i]) & (assignment == target) & unique)
            if not len(candidates):
                raise RuntimeError('Cannot keep duplicates together without altering class counts.')
            j = int(candidates[rng.randrange(len(candidates))])
            old = int(assignment[i])
            assignment[i] = target
            assignment[j] = old
            swaps += 1
    assert Counter(((int(labels[i]), int(assignment[i])) for i in ids)) == before
    assert all((len({int(assignment[i]) for i in group}) == 1 for group in duplicates))
    partitions = [set(np.flatnonzero(assignment == s).tolist()) for s in range(3)]
    assert set.union(*partitions) == set(range(n)) and (not any((partitions[a] & partitions[b] for a, b in [(0, 1), (0, 2), (1, 2)])))
    for key in ['sha256', 'prepared_rgb_sha256']:
        fingerprints = [{rows[i][key] for i in part} for part in partitions]
        assert not any((fingerprints[a] & fingerprints[b] for a, b in [(0, 1), (0, 2), (1, 2)]))
    return (assignment, initial, duplicates, swaps)

def prepare_main(argv=None):
    """prepare_dataset.py: preserved prepare stage implementation."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--input', type=Path, default=Path('output/class_filter/retained_manifest.csv'))
    p.add_argument('--face-manifest', type=Path, default=Path('output/face_detection/face_detection_manifest.csv'))
    p.add_argument('--data-output', type=Path, default=Path('data/processed/prepared_faces'))
    p.add_argument('--output', type=Path, default=Path('output/prepared_dataset'))
    p.add_argument('--seed', type=int, default=4002)
    args = p.parse_args(argv)
    started = time.time()
    if args.data_output.exists() or args.output.exists():
        p.error('Output exists. Use new output paths to preserve previous work.')
    with args.input.open(newline='', encoding='utf-8') as f:
        rows = sorted(list(csv.DictReader(f)), key=lambda r: r['archive_member'])
    assert len(rows) == 73979 and all((r['status'] == 'retained' for r in rows))
    assert len({r['archive_member'] for r in rows}) == len(rows)
    assert all((int(r['estimated_age']) >= 18 and int(r['class_retained_count']) >= 300 for r in rows))
    with args.face_manifest.open(newline='', encoding='utf-8') as f:
        faces = {r['archive_member']: r for r in csv.DictReader(f) if r['status'] == 'retained'}
    assert all((r['archive_member'] in faces for r in rows))
    by_member = {r['archive_member']: i for i, r in enumerate(rows)}
    names = sorted({r['first_name'] for r in rows})
    label_map = {name: i for i, name in enumerate(names)}
    labels = np.array([label_map[r['first_name']] for r in rows], dtype=np.int64)
    n = len(rows)
    args.data_output.mkdir(parents=True)
    args.output.mkdir(parents=True)
    partial = args.data_output / 'faces.h5.partial'
    final = args.data_output / 'faces.h5'
    (args.output / 'RUNNING.json').write_text(json.dumps(dict(complete=False, stage='alignment and normalization', input_images=n), indent=2))
    preview_ids = set(random.Random(args.seed).sample(range(n), 16))
    previews = {}
    seen = set()
    global_min = 1.0
    global_max = -1.0
    print(f'Preparing {n:,} images: 112x112 RGB -> float32 CHW [-1,1].', flush=True)
    with h5py.File(partial, 'w') as h5:
        h5.attrs.update(complete=False, layout='NCHW', color_order='RGB', normalization='uint8 / 127.5 - 1.0', seed=args.seed)
        images = h5.create_dataset('images', shape=(n, 3, 112, 112), dtype='float32', chunks=(1, 3, 112, 112), compression='gzip', compression_opts=4, shuffle=True, fletcher32=True)
        h5.create_dataset('labels', data=labels)
        max_name = max((len(r['filename'].encode()) for r in rows))
        h5.create_dataset('filenames', data=np.array([r['filename'].encode() for r in rows], dtype=f'S{max_name}'))
        h5.create_dataset('class_names', data=np.array(names, dtype=h5py.string_dtype('utf-8')))
        with tarfile.open(args.archive, 'r|gz') as archive:
            for member in archive:
                if member.name not in by_member:
                    continue
                i = by_member[member.name]
                row = rows[i]
                data = archive.extractfile(member).read()
                assert hashlib.sha256(data).hexdigest() == row['sha256'] == faces[member.name]['sha256']
                assert i not in seen
                original = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if original is None:
                    raise RuntimeError(f'Previously decoded input failed: {member.name}')
                detections = json.loads(faces[member.name]['detections_json'])
                assert len(detections) == 1
                landmarks = np.asarray(detections[0]['landmarks'], dtype=np.float32)
                assert landmarks.shape == (5, 2) and np.isfinite(landmarks).all()
                aligned = face_align.norm_crop(original, landmark=landmarks, image_size=112)
                rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
                assert rgb.shape == (112, 112, 3) and rgb.dtype == np.uint8
                if int(rgb.max()) == int(rgb.min()):
                    raise RuntimeError(f"Constant aligned output requires review: {row['filename']}")
                normalized = np.ascontiguousarray((rgb.astype(np.float32) / np.float32(127.5) - np.float32(1)).transpose(2, 0, 1))
                assert normalized.dtype == np.float32 and normalized.shape == (3, 112, 112) and np.isfinite(normalized).all()
                low = float(normalized.min())
                high = float(normalized.max())
                assert low >= -1 and high <= 1
                global_min = min(global_min, low)
                global_max = max(global_max, high)
                restored = np.rint((normalized.transpose(1, 2, 0) + 1) * 127.5).clip(0, 255).astype(np.uint8)
                assert np.array_equal(restored, rgb), 'Normalization did not round-trip to aligned RGB pixels.'
                row['prepared_rgb_sha256'] = hashlib.sha256(rgb.tobytes()).hexdigest()
                row['normalized_tensor_sha256'] = hashlib.sha256(normalized.tobytes()).hexdigest()
                images[i] = normalized
                seen.add(i)
                if i in preview_ids:
                    previews[i] = (cv2.cvtColor(original, cv2.COLOR_BGR2RGB), restored)
                if len(seen) % 1000 == 0:
                    h5.flush()
                    print(f'Prepared {len(seen):,}/{n:,}; elapsed {time.time() - started:.0f}s', flush=True)
        assert len(seen) == n
        print('Image preparation complete. Creating stratified splits with duplicate grouping...', flush=True)
        assignment, initial, duplicates, swaps = prepare_make_splits(rows, labels, args.seed)
        split_names = ['train', 'validation', 'test']
        h5.create_dataset('split', data=assignment)
        split_group = h5.create_group('split_indices')
        for k, name in enumerate(split_names):
            split_group.create_dataset(name, data=np.flatnonzero(assignment == k))
        h5.create_dataset('normalized_sha256', data=np.array([r['normalized_tensor_sha256'].encode() for r in rows], dtype='S64'))
        h5.attrs['complete'] = True
        h5.flush()
    print('Verifying every saved tensor and split assignment...', flush=True)
    with h5py.File(partial, 'r') as h5:
        assert h5['images'].shape == (n, 3, 112, 112) and h5['images'].dtype == np.dtype('float32')
        assert np.array_equal(h5['labels'][:], labels) and np.array_equal(h5['split'][:], assignment)
        for start in range(0, n, 128):
            block = h5['images'][start:start + 128]
            assert np.isfinite(block).all() and block.min() >= -1 and (block.max() <= 1)
            for offset, tensor in enumerate(block):
                i = start + offset
                assert hashlib.sha256(np.ascontiguousarray(tensor).tobytes()).hexdigest() == rows[i]['normalized_tensor_sha256']
    partial.rename(final)
    split_rows = []
    for i, row in enumerate(rows):
        split_rows.append(dict(h5_index=i, filename=row['filename'], archive_member=row['archive_member'], first_name=row['first_name'], label=int(labels[i]), split=split_names[int(assignment[i])], initial_split=split_names[int(initial[i])], duplicate_group_adjusted=bool(initial[i] != assignment[i]), source_sha256=row['source_sha256'], prepared_rgb_sha256=row['prepared_rgb_sha256'], normalized_tensor_sha256=row['normalized_tensor_sha256'], estimated_age=row['estimated_age']))
    fields = list(split_rows[0])
    prepare_write_csv(args.output / 'split_manifest.csv', split_rows, fields)
    for name in split_names:
        prepare_write_csv(args.output / f'{name}_manifest.csv', [r for r in split_rows if r['split'] == name], fields)
    per_name = []
    for name in names:
        indices = np.flatnonzero(labels == label_map[name])
        counts = Counter((split_names[int(assignment[i])] for i in indices))
        assert len(counts) == 3
        per_name.append(dict(first_name=name, label=label_map[name], total=len(indices), **counts))
    prepare_write_csv(args.output / 'split_counts_by_name.csv', per_name, ['first_name', 'label', 'total', 'train', 'validation', 'test'])
    duplicate_rows = []
    for group_id, group in enumerate(duplicates, 1):
        for i in group:
            duplicate_rows.append(dict(group_id=group_id, filename=rows[i]['filename'], first_name=rows[i]['first_name'], source_sha256=rows[i]['sha256'], prepared_rgb_sha256=rows[i]['prepared_rgb_sha256'], split=split_names[int(assignment[i])]))
    prepare_write_csv(args.output / 'duplicate_groups.csv', duplicate_rows, ['group_id', 'filename', 'first_name', 'source_sha256', 'prepared_rgb_sha256', 'split'])
    (args.output / 'label_mapping.json').write_text(json.dumps(label_map, indent=2))
    sheet = Image.new('RGB', (1280, 1120), 'white')
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 14)
    preview_rows = []
    for j, i in enumerate(sorted(previews)):
        x = j % 4 * 320
        y = j // 4 * 280
        original, rgb = previews[i]
        draw.text((x + 5, y + 5), rows[i]['filename'], font=font, fill='black')
        draw.text((x + 5, y + 27), 'Original', font=font, fill='black')
        draw.text((x + 165, y + 27), 'Prepared RGB', font=font, fill='black')
        a = Image.fromarray(original)
        a.thumbnail((144, 200))
        sheet.paste(a, (x + 5, y + 50))
        sheet.paste(Image.fromarray(rgb).resize((144, 144)), (x + 165, y + 50))
        draw.rectangle((x, y, x + 319, y + 279), outline='#bbbbbb')
        preview_rows.append(dict(h5_index=i, filename=rows[i]['filename']))
    sheet.save(args.output / 'alignment_preview.jpg', quality=95)
    prepare_write_csv(args.output / 'preview_manifest.csv', preview_rows, ['h5_index', 'filename'])
    (args.output / '.gitignore').write_text('alignment_preview.jpg\n')
    split_counts = {name: int(np.sum(assignment == k)) for k, name in enumerate(split_names)}
    summary = dict(complete=True, images=n, names=len(names), alignment='five-point ArcFace similarity transform using saved RetinaFace landmarks', image_size=[112, 112], color_order='RGB', tensor_layout='NCHW', dtype='float32', normalization='uint8 / 127.5 - 1.0', range=[global_min, global_max], data_file=str(final.resolve()), file_bytes=final.stat().st_size, compression='lossless gzip level 4 with shuffle and Fletcher32 checksum', split_seed=args.seed, split_proportions=[0.7, 0.15, 0.15], split_counts=split_counts, split_percentages={name: round(100 * c / n, 4) for name, c in split_counts.items()}, stratified_by='first_name', duplicate_groups=len(duplicates), images_in_duplicate_groups=sum(map(len, duplicates)), duplicate_repair_swaps=swaps, cross_name_duplicate_groups=sum((len({rows[i]['first_name'] for i in g}) > 1 for g in duplicates)), class_split_counts_preserved_after_duplicate_repairs=True, source_hash_overlap_between_splits=False, prepared_pixel_hash_overlap_between_splits=False, same_person_different_photos_checked=False, all_saved_tensor_hashes_verified=True, all_images_normalization_roundtrip_verified=True, input_manifest_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(), face_manifest_sha256=hashlib.sha256(args.face_manifest.read_bytes()).hexdigest(), versions={x: importlib.metadata.version(x) for x in ['numpy', 'opencv-python-headless', 'insightface', 'scikit-learn', 'h5py']}, images_removed_in_this_stage=0, previous_exclusions_preserved=True, original_images_modified=False, training_started=False, elapsed_seconds=round(time.time() - started, 1), warning='Estimated-adult eligibility remains unverified. Exact-duplicate grouping does not guarantee distinct identities across splits.')
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    (args.data_output / 'README.md').write_text('# Prepared model inputs\n\nfaces.h5 stores normalized float32 RGB images as NCHW, with labels, filenames, class names, split IDs, and split index arrays. Each image is 3x112x112, already normalized to [-1,1]; do not normalize again. Lossless gzip/shuffle compression and chunk checksums are enabled.\n\nUse scripts/prepared_dataset.py to load a split. See output/prepared_dataset for manifests, label mapping, counts, and verification. These data remain subject to dataset usage terms and are ignored by Git. Original images remain in the source archive.\n', encoding='utf-8')
    (args.output / 'README.md').write_text(f"# Prepared dataset and fixed splits\n\nPrepared all {n:,} Step 3 images, with no further removals. Five-point RetinaFace landmarks were mapped to the standard ArcFace template at 112x112. Images are RGB, float32 CHW, normalized by x/127.5-1. Saved once in a losslessly compressed HDF5 file, not as thousands of duplicate photo/tensor files.\n\nSplits: training {split_counts['train']:,}, validation {split_counts['validation']:,}, test {split_counts['test']:,}. Two-stage name-stratified splitting used seed {args.seed}; same-name swaps kept {len(duplicates)} exact duplicate groups within a single split while preserving every per-name split count. All 100 names occur in each split.\n\nEvery saved tensor was checked for shape, dtype, finite values, [-1,1] range, and exact SHA-256 agreement with its prepared tensor. Normalization was checked to round-trip to the aligned RGB pixels. Filename/index/label links and disjoint split/source/pixel hashes were verified. Exact duplicates are not the same as identity labels: distinct photos of the same person may remain across splits. No claim of verified adulthood is made.\n\nData: data/processed/prepared_faces/faces.h5. Use scripts/prepared_dataset.py. All decisions and source hashes remain traceable in split_manifest.csv. Per-split manifests, per-name counts, duplicate groups, label mapping, and summary.json are included here. alignment_preview.jpg is a small local QA view and is excluded from Git.\n\nNo training or evaluation has been run. Training augmentation and loss-specific sampling are not implemented in this preprocessing stage. Preserve these split assignments for comparisons across loss functions.\n", encoding='utf-8')
    running = args.output / 'RUNNING.json'
    running.write_text(json.dumps(dict(complete=True, summary='summary.json'), indent=2))
    print(json.dumps(summary, indent=2), flush=True)

def finalize_read(p):
    with p.open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))

def finalize_write(p, rows, fields=None):
    with p.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def finalize_deduplicate(rows):
    """Keep the alphabetically first filename in each transitive exact-duplicate group."""
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for key in ['source_sha256', 'prepared_rgb_sha256']:
        first = {}
        for i, row in enumerate(rows):
            if row[key] in first:
                parent[find(i)] = find(first[row[key]])
            else:
                first[row[key]] = i
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[find(i)].append(row)
    keep = set()
    removed = []
    for group in groups.values():
        assert len({r['first_name'] for r in group}) == 1, 'Unresolved duplicate label conflict'
        winner = min(group, key=lambda r: r['filename'])
        keep.add(winner['filename'])
        for row in group:
            if row is not winner:
                removed.append(dict(filename=row['filename'], first_name=row['first_name'], retained_filename=winner['filename'], source_sha256=row['source_sha256'], prepared_rgb_sha256=row['prepared_rgb_sha256'], reason='redundant_exact_duplicate'))
    return [r for r in rows if r['filename'] in keep], sorted(removed, key=lambda r: r['filename'])

def finalize_main(argv=None):
    """finalize_age25.py: preserved finalize stage implementation."""
    parser = argparse.ArgumentParser(description='Apply approved age25 filter to preserved aligned baseline tensors; existing outputs are protected.')
    parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    out = root / 'output/final_age25'
    data = root / 'data/processed/prepared_faces_age25'
    source = root / 'data/processed/prepared_faces/faces.h5'
    if out.exists() or data.exists():
        raise RuntimeError('Output exists; preserving previous run')
    started = time.time()
    old = finalize_read(root / 'output/prepared_dataset/split_manifest.csv')
    ages = finalize_read(root / 'output/age_filter/age_filter_manifest.csv')
    exclusions = finalize_read(root / 'data/user_review_exclusions.csv')
    exclusions += finalize_read(root / 'data/additional_exclusions.csv')
    excluded_hashes = {r['sha256'] for r in exclusions}
    lookup = {r['filename']: r for r in ages}
    input_hash = hashlib.sha256((root / 'output/prepared_dataset/split_manifest.csv').read_bytes()).hexdigest()
    selected = [dict(r, previous_h5_index=r['h5_index']) for r in old if int(r['estimated_age']) >= 25 and float(lookup[r['filename']]['retinaface_score']) >= 0.5 and r['source_sha256'] not in excluded_hashes]
    selected, duplicate_removals = finalize_deduplicate(selected)
    assert len(duplicate_removals) == 139
    counts = Counter((r['first_name'] for r in selected))
    eligible = {n for n, c in counts.items() if c >= 300}
    selected = [r for r in selected if r['first_name'] in eligible]
    assert len(selected) == 64398 and len(eligible) == 100
    assert not any((r['source_sha256'] in excluded_hashes for r in selected))
    for r in selected:
        assert r['source_sha256'] == lookup[r['filename']]['sha256']
    out.mkdir(parents=True)
    data.mkdir(parents=True)
    partial = data / 'faces.h5.partial'
    final = data / 'faces.h5'
    (out / 'RUNNING.json').write_text(json.dumps(dict(stage='copying verified tensors', complete=False)))
    splits = ['train', 'validation', 'test']
    assignment = np.array([splits.index(r['split']) for r in selected], dtype=np.int8)
    labels = np.array([int(r['label']) for r in selected], dtype=np.int64)
    print(f'Writing {len(selected):,} age-25 tensors; preserving split assignments.', flush=True)
    with h5py.File(source, 'r') as src, h5py.File(partial, 'w') as dst:
        assert src.attrs['complete'] and len(src['images']) == len(old)
        source_names = src['filenames'][:]
        source_labels = src['labels'][:]
        source_splits = src['split'][:]
        source_hashes = src['normalized_sha256'][:]
        for k, v in src.attrs.items():
            dst.attrs[k] = v
        dst.attrs.update(complete=False, estimated_age_cutoff=25, detection_threshold=0.5, split_strategy='Preserved age18 partition assignments; subset only')
        images = dst.create_dataset('images', shape=(len(selected), 3, 112, 112), dtype='float32', chunks=(1, 3, 112, 112), compression='gzip', compression_opts=4, shuffle=True, fletcher32=True)
        for i, r in enumerate(selected):
            j = int(r['previous_h5_index'])
            assert source_names[j].decode() == r['filename'] and int(source_labels[j]) == int(r['label']) and (int(source_splits[j]) == int(assignment[i]))
            assert source_hashes[j].decode() == r['normalized_tensor_sha256']
            assert src['images'].chunks == images.chunks and src['images'].compression == images.compression and src['images'].compression_opts == images.compression_opts and src['images'].shuffle == images.shuffle and src['images'].fletcher32 == images.fletcher32
            mask, chunk = src['images'].id.read_direct_chunk((j, 0, 0, 0))
            images.id.write_direct_chunk((i, 0, 0, 0), chunk, filter_mask=mask)
            r['h5_index'] = i
            if (i + 1) % 10000 == 0:
                print(f'Copied {i + 1:,}/{len(selected):,}; {time.time() - started:.0f}s', flush=True)
        dst.create_dataset('labels', data=labels)
        dst.create_dataset('filenames', data=np.array([r['filename'].encode() for r in selected], dtype=src['filenames'].dtype))
        dst.create_dataset('class_names', data=src['class_names'][:])
        dst.create_dataset('split', data=assignment)
        dst.create_dataset('normalized_sha256', data=np.array([r['normalized_tensor_sha256'].encode() for r in selected], dtype='S64'))
        g = dst.create_group('split_indices')
        for k, s in enumerate(splits):
            g.create_dataset(s, data=np.flatnonzero(assignment == k))
        dst.flush()
    print('Verifying every stored tensor and duplicate separation...', flush=True)
    with h5py.File(partial, 'r+') as f:
        for start in range(0, len(selected), 128):
            for offset, tensor in enumerate(f['images'][start:start + 128]):
                assert tensor.shape == (3, 112, 112) and tensor.dtype == np.float32 and np.isfinite(tensor).all() and tensor.min() >= -1 and tensor.max() <= 1
                assert hashlib.sha256(tensor.tobytes()).hexdigest() == selected[start + offset]['normalized_tensor_sha256']
        assert np.array_equal(f['labels'][:], labels) and np.array_equal(f['split'][:], assignment)
        for key in ['source_sha256', 'prepared_rgb_sha256']:
            sets = [{r[key] for r in selected if r['split'] == s} for s in splits]
            assert all((not sets[a] & sets[b] for a, b in [(0, 1), (0, 2), (1, 2)]))
        f.attrs['complete'] = True
    partial.rename(final)
    finalize_write(out / 'split_manifest.csv', selected)
    finalize_write(out / 'duplicate_removals.csv', duplicate_removals)
    for s in splits:
        finalize_write(out / f'{s}_manifest.csv', [r for r in selected if r['split'] == s])
    name_rows = []
    for n in sorted(counts):
        name_rows.append(dict(first_name=n, total=counts[n], **{s: sum((r['first_name'] == n and r['split'] == s for r in selected)) for s in splits}))
    assert all((min((r[s] for s in splits)) > 0 for r in name_rows))
    finalize_write(out / 'split_counts_by_name.csv', name_rows)
    decisions = []
    retained_filenames = {r['filename'] for r in selected}
    duplicate_filenames = {r['filename'] for r in duplicate_removals}
    for a in ages:
        r = dict(a)
        keep = r['filename'] in retained_filenames
        r.update(age_cutoff=25, status='retained' if keep else 'excluded', reason='explicit_user_exclusion_label_conflict' if r['sha256'] in excluded_hashes else ('redundant_exact_duplicate' if r['filename'] in duplicate_filenames else ('estimated_age_at_least_25' if keep else 'estimated_age_under_25')))
        decisions.append(r)
    finalize_write(out / 'age_filter_manifest.csv', decisions)
    kept_names = {r['filename'] for r in selected}
    removed = [dict(r, removal_reason='explicit_user_exclusion_label_conflict' if r['source_sha256'] in excluded_hashes else ('redundant_exact_duplicate' if r['filename'] in duplicate_filenames else 'estimated_age_under_25')) for r in old if r['filename'] not in kept_names]
    finalize_write(out / 'removed_from_age18.csv', removed)
    label_map = json.loads((root / 'output/prepared_dataset/label_mapping.json').read_text())
    (out / 'label_mapping.json').write_text(json.dumps(label_map, indent=2))
    duplicate_rows = [r for r in finalize_read(root / 'output/prepared_dataset/duplicate_groups.csv') if r['filename'] in kept_names]
    grouped = Counter((r['group_id'] for r in duplicate_rows))
    duplicate_rows = [r for r in duplicate_rows if grouped[r['group_id']] > 1]
    finalize_write(out / 'duplicate_groups.csv', duplicate_rows, ['group_id', 'filename', 'first_name', 'source_sha256', 'prepared_rgb_sha256', 'split'])
    conflicts = [r for r in duplicate_rows if len({x['first_name'] for x in duplicate_rows if x['group_id'] == r['group_id']}) > 1]
    finalize_write(out / 'label_conflicts.csv', conflicts, ['group_id', 'filename', 'first_name', 'source_sha256', 'prepared_rgb_sha256', 'split'])
    loader_checks = {}
    for s in splits:
        loader = PreparedFaces(final, s)
        assert len(loader) == sum(assignment == splits.index(s))
        for i in [0, len(loader) // 2, len(loader) - 1]:
            x, y = loader[i]
            assert x.shape == (3, 112, 112) and x.dtype == np.float32 and (0 <= y < 100)
        loader_checks[s] = len(loader)
        loader.close()
    digest = hashlib.sha256()
    with final.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(b)
    split_counts = Counter((r['split'] for r in selected))
    summary = dict(complete=True, images=len(selected), names=len(eligible), age_cutoff=25, detection_threshold=0.5, minimum_class_size=300, removed_from_previous_dataset=len(removed), excluded_from_face_retained=len(ages) - len(selected), minimum_class_count=min(counts.values()), split_counts=dict(split_counts), split_percentages={s: round(100 * split_counts[s] / len(selected), 3) for s in splits}, split_strategy='Preserve previous assignments; do not repartition', data_file=str(final), file_bytes=final.stat().st_size, hdf5_sha256=digest.hexdigest(), input_split_manifest_sha256=input_hash, source_hdf5=str(source), prior_user_exclusions_reintroduced=False, all_tensors_hash_verified=True, duplicate_hash_overlap_between_splits=False, label_mapping_changed=False, model_inference_rerun=False, manual_visual_cleanup_performed=True, additional_manual_exclusions=2, duplicate_copies_removed=len(duplicate_removals), cross_name_conflict_images=len(conflicts), duplicate_groups=len({r['group_id'] for r in duplicate_rows}), images_in_duplicate_groups=len(duplicate_rows), redundant_duplicate_copies=len(duplicate_rows)-len({r['group_id'] for r in duplicate_rows}), loader_verified=loader_checks, elapsed_seconds=round(time.time() - started, 1), warning='Estimated age is not verified adulthood. Children and artificial images can remain. Different photos of the same individual may cross splits.')
    assert len(removed) == 9581 and sum(split_counts.values()) == 64398
    assert hashlib.sha256((root / 'output/prepared_dataset/split_manifest.csv').read_bytes()).hexdigest() == input_hash
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    (out / 'RUNNING.json').unlink()
    handoff = f"# Active preprocessing dataset: estimated age 25+\n\nGroup-approved settings: estimated age >=25; RetinaFace confidence >=0.5; minimum 300 images/name. Retained {len(selected):,} images across 100 names; removed {len(removed):,} from the earlier age18 dataset. Mike_311.png and Steve_550.png were explicitly excluded by user instruction because their identical aligned image had conflicting labels. Additional exclusions are recorded in data/additional_exclusions.csv. 139 additional redundant copies were removed by exact source/aligned-pixel hashing; see duplicate_removals.csv. No broader manual review or cartoon detection was performed. These are model-estimated ages; children and artificial faces can remain.\n\nTrain {split_counts['train']:,}; validation {split_counts['validation']:,}; test {split_counts['test']:,}. Original split assignments were preserved, so percentages are approximately 70/15/15 rather than a new stratification. Every name remains in all three splits. Exact source/pixel duplicates remain within one partition; different photos of the same person are not identity-disjoint. No remaining cross-name exact-duplicate conflicts were found. 139 redundant exact-duplicate copies were removed, keeping one deterministic representative per group; labels were not guessed or changed.\n\nActive data: data/processed/prepared_faces_age25/faces.h5. RGB float32 NCHW, shape (64398,3,112,112), normalized to [-1,1]. Do not normalize again. Labels remain 0-99 with the unchanged label_mapping.json. Use scripts/prepared_dataset.py:\n\n```python\nfrom prepared_dataset import PreparedFaces\ntrain = PreparedFaces('data/processed/prepared_faces_age25/faces.h5', 'train')\nimage, label = train[0]\ntrain.close()\n```\n\nReproduce from the preserved earlier prepared dataset: `python scripts/preprocess.py finalize`. Existing output paths are protected against overwrite. All source tensors and every saved tensor were hash-verified; loader checks and duplicate separation passed. Earlier age18 files remain as historical inputs, not active training data. No original photos were modified.\n\nThe age cutoff changed from MI2's 18 to 25 by group decision after threshold comparisons. Detection stayed at 0.5. Confidence/age filtering does not certify photographic realism or adulthood. No model training or evaluation was performed.\n"
    (out / 'HANDOFF.md').write_text(handoff, encoding='utf-8')
    (out / 'README.md').write_text(handoff, encoding='utf-8')
    (root / 'data/processed/active_dataset.json').write_text(json.dumps(dict(data_file='data/processed/prepared_faces_age25/faces.h5', report='output/final_age25/HANDOFF.md', age_cutoff=25, detection_threshold=0.5, images=64398, names=100), indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def main(argv=None):
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    handlers = {'detect': detect_main, 'estimate-age': estimate_age_main,
                'filter-classes': filter_classes_main, 'prepare': prepare_main,
                'finalize': finalize_main}
    parser = argparse.ArgumentParser(description=__doc__, epilog='Use STAGE --help for stage options. Run from the repository root. Existing outputs are protected.')
    parser.add_argument('stage', choices=list(handlers))
    if not argv or argv[0].startswith('-'):
        parser.parse_args(argv)
        return
    args = parser.parse_args(argv[:1])
    handlers[args.stage](argv[1:])

if __name__ == '__main__':
    main()
