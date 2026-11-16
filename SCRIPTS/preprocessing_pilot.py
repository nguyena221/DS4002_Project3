"""Annie Nguyen | DS4002 Project 3 | MI2 preprocessing pilot.

Samples 10 images per name with a fixed seed, without modifying the archive.
Uses buffalo_l's SCRFD detector (a documented difference from MI2's RetinaFace)
and age estimator. Retains exactly one detected face with estimated age >=18,
aligns it to 112x112 RGB, and saves a float32 CHW tensor in [-1,1].
This is NOT the final dataset: the >=300 class threshold and 70/15/15 splits
must be applied after full screening. Estimated ages do not verify adulthood.
Run from the repository root using the project's Python environment.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import tarfile
import time
from collections import Counter

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.utils import face_align


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('OUTPUT/preprocessing_pilot'))
    parser.add_argument('--model-root', type=Path, default=Path('models'))
    parser.add_argument('--per-name', type=int, default=10)
    parser.add_argument('--seed', type=int, default=4002)
    parser.add_argument('--det-thresh', type=float, default=0.5)
    parser.add_argument('--det-size', type=int, default=160,
                        help='Detector input side; 160 tested for these tightly cropped images.')
    args = parser.parse_args()
    if args.per_name < 1 or not 0 < args.det_thresh < 1 or args.det_size < 32 or args.det_size % 32:
        parser.error('per-name must be positive and det-thresh must be between 0 and 1')
    if not args.archive.is_file():
        parser.error(f'Archive not found: {args.archive}')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output folder is not empty. Use a new --output folder to preserve prior results.')
    started = time.time()
    # Reservoir sampling gives every image within a name equal selection probability.
    # Only selected image bytes are retained in memory; no archive paths are extracted.
    rng = random.Random(args.seed)
    samples = {}
    counts = Counter()
    print('Sampling the archive; this can take a minute...', flush=True)
    with tarfile.open(args.archive, 'r|gz') as archive:
        for member in archive:
            if not member.isfile() or not member.name.lower().endswith('.png'):
                continue
            filename = member.name.rsplit('/', 1)[-1]
            name, number = Path(filename).stem.rsplit('_', 1)
            if not name.isalpha() or not number.isdigit():
                raise ValueError(f'Unexpected filename: {filename}')
            counts[name] += 1
            reservoir = samples.setdefault(name, [])
            slot = len(reservoir) if len(reservoir) < args.per_name else rng.randrange(counts[name])
            if slot < args.per_name:
                item = (member.name, filename, archive.extractfile(member).read())
                if slot == len(reservoir):
                    reservoir.append(item)
                else:
                    reservoir[slot] = item
    total = sum(map(len, samples.values()))
    print(f'Selected {total:,} images from {len(samples)} names. Loading models...', flush=True)
    app = FaceAnalysis(name='buffalo_l', root=str(args.model_root.resolve()),
                       allowed_modules=['detection', 'genderage'],
                       providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=-1, det_size=(args.det_size, args.det_size), det_thresh=args.det_thresh)
    args.output.mkdir(parents=True, exist_ok=True)
    crops = args.output / 'crops'
    tensors = args.output / 'tensors'
    crops.mkdir()
    tensors.mkdir()
    fields = ['archive_member', 'filename', 'first_name', 'sha256', 'face_count',
              'detection_score', 'estimated_age', 'status', 'reason', 'crop_path', 'tensor_path']
    reasons = Counter()
    by_name = Counter()
    done = 0
    with (args.output / 'screening_manifest.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for name in sorted(samples):
            for member, filename, data in sorted(samples[name]):
                row = dict.fromkeys(fields, '')
                row.update(archive_member=member, filename=filename, first_name=name,
                           sha256=hashlib.sha256(data).hexdigest(), status='excluded')
                image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    row['reason'] = 'decode_failed'
                else:
                    faces = app.get(image)
                    row['face_count'] = len(faces)
                    if len(faces) != 1:
                        row['reason'] = 'no_face' if not faces else 'multiple_faces'
                    else:
                        face = faces[0]
                        row['detection_score'] = float(face.det_score)
                        age = face.get('age')
                        if age is None or not np.isfinite(age):
                            row['reason'] = 'age_unavailable'
                        else:
                            row['estimated_age'] = float(age)
                            if age < 18:
                                row['reason'] = 'estimated_age_under_18'
                            elif face.kps is None or not np.isfinite(face.kps).all():
                                row['reason'] = 'landmarks_unavailable'
                            else:
                                # Landmark alignment avoids stretching a rectangular raw image.
                                aligned = face_align.norm_crop(image, landmark=face.kps, image_size=112)
                                rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
                                tensor = np.ascontiguousarray((rgb.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1))
                                assert tensor.shape == (3, 112, 112)
                                assert np.isfinite(tensor).all() and tensor.min() >= -1 and tensor.max() <= 1
                                crop_path = crops / filename
                                tensor_path = tensors / (Path(filename).stem + '.npy')
                                if not cv2.imwrite(str(crop_path), aligned):
                                    raise OSError(f'Could not save {crop_path}')
                                np.save(tensor_path, tensor, allow_pickle=False)
                                row.update(status='retained', reason='estimated_age_at_least_18',
                                           crop_path=crop_path.relative_to(args.output).as_posix(),
                                           tensor_path=tensor_path.relative_to(args.output).as_posix())
                                by_name[name] += 1
                reasons[row['reason']] += 1
                writer.writerow(row)
                f.flush()
                done += 1
                if done % 50 == 0:
                    print(f'Processed {done}/{total}', flush=True)
    with (args.output / 'name_counts.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['first_name', 'raw_count', 'pilot_sampled', 'pilot_retained'])
        writer.writerows((n, counts[n], len(samples[n]), by_name[n]) for n in sorted(samples))
    versions = {p: importlib.metadata.version(p) for p in ['insightface','onnxruntime','numpy','opencv-python-headless']}
    summary = dict(pilot_only=True, archive=str(args.archive.resolve()), seed=args.seed,
                   sample_per_name=args.per_name, sampled=total, retained=sum(by_name.values()),
                   outcomes=dict(reasons), estimated_age_cutoff=18, detection_threshold=args.det_thresh,
                   detection_size=[args.det_size,args.det_size], detector='SCRFD in buffalo_l; differs from MI2 RetinaFace',
                   provider='CPUExecutionProvider', model_root=str(args.model_root.resolve()),
                   normalization='RGB float32 /127.5 -1; CHW (3,112,112); do not normalize again',
                   class_minimum_applied=False, splits_created=False, package_versions=versions,
                   elapsed_seconds=round(time.time()-started, 1))
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
    print(f'Pilot complete. Review: {args.output.resolve()}', flush=True)


if __name__ == '__main__':
    main()
