"""Stream the original archive without extracting or modifying images.
Run: python scripts/explore_data.py --archive PATH --output OUTPUT_DIRECTORY
Requires numpy, Pillow, matplotlib.
"""
import argparse
import collections
import csv
import io
import json
from pathlib import Path
import tarfile

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', default='data/raw/Names100Dataset.tar.gz')
    parser.add_argument('--output', default='output/eda')
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists():
        parser.error('Output exists; choose a new --output directory to preserve earlier results.')
    out.mkdir(parents=True, exist_ok=True)
    counts = collections.Counter()
    values = []
    failures = []
    with (out / 'image_brightness.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'first_name', 'mean_brightness_0_255'])
        with tarfile.open(args.archive, 'r|gz') as archive:
            for member in archive:
                if not member.isfile() or not member.name.lower().endswith('.png'):
                    continue
                filename = member.name.rsplit('/', 1)[-1]
                name = filename.rsplit('_', 1)[0]
                counts[name] += 1
                try:
                    with Image.open(io.BytesIO(archive.extractfile(member).read())) as image:
                        # BT.601-style luma on encoded RGB; not physical luminance.
                        channel_means = np.asarray(image.convert('RGB'), dtype=np.float64).mean(axis=(0, 1))
                        brightness = float(channel_means @ np.array([0.299, 0.587, 0.114]))
                    values.append(brightness)
                    writer.writerow([filename, name, f'{brightness:.6f}'])
                except Exception as error:
                    failures.append({'filename': filename, 'error': str(error)})
                if sum(counts.values()) % 10000 == 0:
                    print(f'Processed {sum(counts.values()):,} images', flush=True)
    names = sorted(counts)
    with (out / 'name_counts.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['first_name', 'image_count'])
        writer.writerows((name, counts[name]) for name in names)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 4, figsize=(14, 8), sharex=True)
    for ax, group in zip(axes, np.array_split(names, 4)):
        ax.barh(group, [counts[n] for n in group], color='#347C98', height=0.72)
        ax.invert_yaxis()
        ax.set_xlim(0, max(counts.values()) * 1.1)
        ax.set_xticks([0, 400, 800])
        ax.set_xlabel('Number of images')
        ax.tick_params(axis='y', labelsize=9)
        ax.set_axisbelow(True)
        ax.grid(axis='x', alpha=0.18)
    fig.suptitle('Raw image counts are balanced across all 100 first names', fontsize=17, y=0.99)
    fig.text(0.5, 0.945, f'{sum(counts.values()):,} images | {len(names)} names | '
             f'{min(counts.values()):,}-{max(counts.values()):,} images per name', ha='center', color='#555555')
    fig.tight_layout(rect=(0, 0.025, 1, 0.92))
    fig.savefig(out / '01_images_per_name.png', dpi=240, facecolor='white')
    plt.close(fig)
    v = np.asarray(values)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(v, bins=np.linspace(0, 255, 52), color='#347C98', edgecolor='white', linewidth=0.5)
    ax.axvline(np.median(v), color='#B45232', linewidth=2,
               label=f'Median: {np.median(v):.1f}')
    ax.set(xlim=(0, 255), xlabel='Mean image brightness (0 = black; 255 = white)',
           ylabel='Number of images')
    ax.set_title('Distribution of mean image brightness', loc='left', fontsize=17, pad=30)
    ax.text(0, 1.025, f'All {len(v):,} successfully decoded images | Before adult or quality screening',
            transform=ax.transAxes, color='#555555')
    ax.legend(frameon=False)
    ax.set_axisbelow(True)
    ax.grid(axis='y', alpha=0.18)
    fig.tight_layout()
    fig.savefig(out / '02_image_brightness.png', dpi=240, facecolor='white')
    plt.close(fig)
    summary = {'archive': str(Path(args.archive).resolve()), 'image_count': sum(counts.values()),
               'name_count': len(counts), 'min_images_per_name': min(counts.values()),
               'max_images_per_name': max(counts.values()), 'decoded_images': len(v),
               'brightness_mean': float(v.mean()), 'brightness_median': float(np.median(v)),
               'brightness_sd': float(v.std()),
               'brightness_percentiles': dict(zip(['min','p05','p25','p75','p95','max'],
                   map(float, np.percentile(v, [0,5,25,75,95,100])))),
               'brightness_definition': 'Mean of 0.299R + 0.587G + 0.114B over all pixels; encoded RGB, 0-255.',
               'decode_failures': failures}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
