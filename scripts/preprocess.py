"""Preprocess the Names100 archive into one aligned, filtered, split face dataset.

Pipeline (each step is one function, run in order by main):
  1. detect_face        RetinaFace (UniFace): keep images with exactly one face
  2. estimate_age       InsightFace buffalo_l age model on the detected box
  3. align_face         five-point ArcFace alignment to 112x112 RGB
  4. remove_duplicates  drop exact duplicate copies (source or aligned pixels)
  5. filter_classes     keep first names with at least MIN_PER_NAME images
  6. split_dataset      stratified 70/15/15 train/validation/test split

Outputs: data/processed/faces.h5 and data/processed/manifest.csv.
Run from the repository root: python scripts/preprocess.py
"""
import argparse, csv, hashlib, io, json, tarfile, time
from collections import Counter, defaultdict
from pathlib import Path

import cv2, h5py, numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split

AGE_CUTOFF = 25          # estimated age must be >= this
DETECTION_CONFIDENCE = 0.5
MIN_PER_NAME = 300
SEED = 4002
SPLITS = ('train', 'validation', 'test')
EXPECTED_IMAGES = 80000
EXPECTED_NAMES = 100

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = ROOT / 'data/raw/Names100Dataset.tar.gz'
DEFAULT_EXCLUSIONS = ROOT / 'data/exclusions.csv'
DEFAULT_OUTPUT = ROOT / 'data/processed'
DEFAULT_AGE_MODEL = Path.home() / '.insightface/models/buffalo_l/genderage.onnx'


def load_exclusions(path):
    """SHA-256 hashes of images the group manually excluded."""
    with path.open(newline='', encoding='utf-8') as f:
        return {r['sha256'] for r in csv.DictReader(f) if r['decision'] == 'exclude'}


def iter_archive(path):
    """Yield (filename, first_name, png_bytes) for each PNG, streamed from the tar.gz."""
    with tarfile.open(path, 'r|gz') as archive:
        for member in archive:
            if not member.isfile() or not member.name.lower().endswith('.png'):
                continue
            filename = member.name.rsplit('/', 1)[-1]
            name, _, number = Path(filename).stem.rpartition('_')
            if not name.isalpha() or not number.isdigit():
                continue
            yield filename, name, archive.extractfile(member).read()


def load_detector():
    import onnxruntime as ort
    from uniface.constants import RetinaFaceWeights
    from uniface.detection import RetinaFace
    detector = RetinaFace(model_name=RetinaFaceWeights.MNET_V2, input_size=(160, 160),
                          confidence_threshold=DETECTION_CONFIDENCE, providers=['CPUExecutionProvider'])
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    detector.session = ort.InferenceSession(detector._model_path, sess_options=opts,
                                            providers=['CPUExecutionProvider'])
    return detector


def load_age_model(path):
    import onnxruntime as ort
    from insightface.model_zoo.attribute import Attribute
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=opts, providers=['CPUExecutionProvider'])
    return Attribute(model_file=str(path), session=session)


def detect_face(detector, bgr):
    """Return (bbox, landmarks, score) when the image has exactly one valid face, else None."""
    faces = detector.detect(bgr)
    if len(faces) != 1:
        return None
    face = faces[0]
    box = np.asarray(face.bbox, dtype=np.float32)
    landmarks = None if face.landmarks is None else np.asarray(face.landmarks, dtype=np.float32)
    if box.shape != (4,) or not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
        return None
    if landmarks is None or landmarks.shape != (5, 2) or not np.isfinite(landmarks).all():
        return None
    return box, landmarks, float(face.confidence)


def estimate_age(age_model, bgr, box):
    from insightface.app.common import Face
    _, age = age_model.get(bgr, Face(bbox=box))
    return int(age)


def align_face(bgr, landmarks):
    """Five-point ArcFace alignment; returns a 112x112 RGB uint8 array."""
    from insightface.utils import face_align
    aligned = face_align.norm_crop(bgr, landmark=landmarks, image_size=112)
    return cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)


def process_archive(archive, excluded, age_model_path):
    """Steps 1-3 over the whole archive. Returns one record per image that passes all three."""
    detector = load_detector()
    age_model = load_age_model(age_model_path)
    records, stats, seen, started = [], Counter(), set(), time.time()
    for filename, name, data in iter_archive(archive):
        if filename in seen:
            raise ValueError(f'Duplicate filename: {filename}')
        seen.add(filename)
        stats['images'] += 1
        sha = hashlib.sha256(data).hexdigest()
        if sha in excluded:
            stats['manual_exclusion'] += 1
        else:
            try:
                with Image.open(io.BytesIO(data)) as image:
                    bgr = cv2.cvtColor(np.asarray(image.convert('RGB')), cv2.COLOR_RGB2BGR)
            except (UnidentifiedImageError, OSError, ValueError):
                stats['unreadable'] += 1
                bgr = None
            detection = None if bgr is None else detect_face(detector, bgr)
            if bgr is not None and detection is None:
                stats['not_single_face'] += 1
            elif detection is not None:
                box, landmarks, score = detection
                age = estimate_age(age_model, bgr, box)
                if age < AGE_CUTOFF:
                    stats['under_age_cutoff'] += 1
                else:
                    rgb = align_face(bgr, landmarks)
                    records.append(dict(filename=filename, first_name=name, source_sha256=sha,
                                        estimated_age=age, detection_score=round(score, 4), rgb=rgb,
                                        aligned_sha256=hashlib.sha256(rgb.tobytes()).hexdigest()))
        if stats['images'] % 1000 == 0:
            print(f"{stats['images']:,}/{EXPECTED_IMAGES:,} images; {len(records):,} passing; "
                  f'{time.time() - started:.0f}s', flush=True)
    stats['passing'] = len(records)
    return records, stats


def remove_duplicates(records):
    """Keep the alphabetically first filename of each exact-duplicate group (source or aligned pixels)."""
    parent = list(range(len(records)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for key in ('source_sha256', 'aligned_sha256'):
        first = {}
        for i, r in enumerate(records):
            if r[key] in first:
                parent[find(i)] = find(first[r[key]])
            else:
                first[r[key]] = i
    groups = defaultdict(list)
    for i, r in enumerate(records):
        groups[find(i)].append(r)
    keep = set()
    for group in groups.values():
        if len({r['first_name'] for r in group}) > 1:
            raise ValueError('Identical image appears under different names: ' + ', '.join(r['filename'] for r in group))
        keep.add(min(group, key=lambda r: r['filename'])['filename'])
    return [r for r in records if r['filename'] in keep]


def filter_classes(records):
    counts = Counter(r['first_name'] for r in records)
    return [r for r in records if counts[r['first_name']] >= MIN_PER_NAME]


def split_dataset(labels):
    """Stratified 70/15/15 split of label array; returns an array of split indices (0/1/2)."""
    ids = np.arange(len(labels))
    train, hold = train_test_split(ids, train_size=0.7, stratify=labels, random_state=SEED)
    validation, test = train_test_split(hold, test_size=0.5, stratify=labels[hold], random_state=SEED + 1)
    assignment = np.empty(len(labels), dtype=np.int8)
    for split, indices in enumerate((train, validation, test)):
        assignment[indices] = split
    return assignment


def write_outputs(records, output):
    records = sorted(records, key=lambda r: r['filename'])
    names = sorted({r['first_name'] for r in records})
    label_of = {name: i for i, name in enumerate(names)}
    labels = np.array([label_of[r['first_name']] for r in records], dtype=np.int64)
    assignment = split_dataset(labels)
    n = len(records)
    output.mkdir(parents=True, exist_ok=True)
    final, partial = output / 'faces.h5', output / 'faces.h5.partial'
    with h5py.File(partial, 'w') as h5:
        h5.attrs.update(complete=False, layout='NCHW', color_order='RGB', normalization='uint8 / 127.5 - 1.0',
                        age_cutoff=AGE_CUTOFF, detection_confidence=DETECTION_CONFIDENCE, seed=SEED)
        images = h5.create_dataset('images', shape=(n, 3, 112, 112), dtype='float32', chunks=(1, 3, 112, 112),
                                   compression='gzip', compression_opts=4, shuffle=True, fletcher32=True)
        for i, r in enumerate(records):
            images[i] = (r['rgb'].astype(np.float32) / np.float32(127.5) - np.float32(1)).transpose(2, 0, 1)
        h5.create_dataset('labels', data=labels)
        h5.create_dataset('split', data=assignment)
        h5.create_dataset('filenames', data=np.array([r['filename'].encode() for r in records]))
        h5.create_dataset('class_names', data=np.array(names, dtype=h5py.string_dtype('utf-8')))
        group = h5.create_group('split_indices')
        for k, split in enumerate(SPLITS):
            group.create_dataset(split, data=np.flatnonzero(assignment == k))
        h5.attrs['complete'] = True
    partial.rename(final)
    fields = ['h5_index', 'filename', 'first_name', 'label', 'split', 'estimated_age', 'detection_score', 'source_sha256']
    with (output / 'manifest.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for i, r in enumerate(records):
            writer.writerow(dict(r, h5_index=i, label=int(labels[i]), split=SPLITS[int(assignment[i])]))
    return labels, assignment


def verify(output, labels, assignment):
    with h5py.File(output / 'faces.h5', 'r') as h5:
        assert h5.attrs['complete'] and h5['images'].shape == (len(labels), 3, 112, 112)
        assert np.array_equal(h5['labels'][:], labels) and np.array_equal(h5['split'][:], assignment)
        for start in range(0, len(labels), 256):
            block = h5['images'][start:start + 256]
            assert np.isfinite(block).all() and block.min() >= -1 and block.max() <= 1
    counts = Counter(zip(labels.tolist(), assignment.tolist()))
    assert all(counts[(label, s)] > 0 for label in set(labels.tolist()) for s in range(3)), 'name missing from a split'
    return {s: int((assignment == k).sum()) for k, s in enumerate(SPLITS)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--archive', type=Path, default=DEFAULT_ARCHIVE)
    p.add_argument('--exclusions', type=Path, default=DEFAULT_EXCLUSIONS)
    p.add_argument('--age-model', type=Path, default=DEFAULT_AGE_MODEL)
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = p.parse_args(argv)
    for path in (args.archive, args.exclusions, args.age_model):
        if not path.is_file():
            p.error(f'Missing file: {path}')
    if (args.output / 'faces.h5').exists() or (args.output / 'manifest.csv').exists():
        p.error('Outputs already exist in ' + str(args.output) + '; move or delete them first.')

    started = time.time()
    records, stats = process_archive(args.archive, load_exclusions(args.exclusions), args.age_model)
    assert stats['images'] == EXPECTED_IMAGES, f"Expected {EXPECTED_IMAGES} images, found {stats['images']}"
    records = remove_duplicates(records)
    stats['after_duplicates'] = len(records)
    records = filter_classes(records)
    stats['final'] = len(records)
    labels, assignment = write_outputs(records, args.output)
    stats['splits'] = verify(args.output, labels, assignment)
    stats['names'] = len(set(labels.tolist()))
    stats['seconds'] = round(time.time() - started)
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
