# Do You Look Like Your Name?

DS 4002, Section 001 | Project 3 | EAZ Inc.

**Team:** Eli Boyden (group leader; training), Zilan Saadi (evaluation), Annie Nguyen (preprocessing).

**Research question:** Can fine-tuned machine learning models predict whether two adults share a given name from facial images alone?

The planned pipeline screens Names100 images, prepares model inputs, and compares Triplet Margin, ArcFace, and Supervised Contrastive losses using a pretrained IResNet-100 backbone. The evaluation target is triplet comparison accuracy above 50%. Preprocessing, training, and final evaluation have not yet been implemented here. The exploratory figures have been generated from the full raw archive.

## 1. Software and platform

- Initial exploratory analysis: Python on Windows.
- Exploratory packages: NumPy, Pillow, and Matplotlib (see `requirements-eda.txt`).
- Planned preprocessing: InsightFace, ONNX Runtime, and image-processing utilities; exact versions and hardware will be recorded when the pilot is configured.
- Planned modeling: PyTorch and pytorch-metric-learning on UVA LooseLeaf.
- GPU configuration and the model checkpoint must be documented before training.

The InsightFace model package and the PyTorch training backbone serve different purposes. The planned `buffalo_l` preprocessing package should not be assumed to supply the team's IResNet-100 training checkpoint.

## 2. Repository contents and documentation map

```text
DS4002_Project3/
|-- README.md
|-- LICENSE.md
|-- requirements-eda.txt
|-- .gitignore
|-- SCRIPTS/
|   |-- README.md
|   `-- 01_explore_raw_data.py
|-- DATA/
|   |-- README.md
|   |-- raw/                 # local images/archive; ignored by Git
|   `-- processed/           # local processed images/tensors; ignored by Git
`-- OUTPUT/
    |-- README.md
    |-- eda/
    |   |-- 01_images_per_name.png
    |   |-- 02_image_brightness.png
    |   |-- name_counts.csv
    |   |-- summary.json
    |   `-- MI2_captions.md
    `-- preprocessing_plan.md
```

`DATA/README.md` supplies access instructions, provenance, usage terms, ethics, and a data dictionary. `SCRIPTS/README.md` documents execution order. `OUTPUT/preprocessing_plan.md` tracks implementation decisions and deviations from MI2. Final scripts and results will be added as the analysis is completed.

## 3. Reproducing the current exploratory results

1. Obtain `Names100Dataset.tar.gz` from the original dataset source [3], review its use terms, and keep the archive locally. It is not included in this repository.
2. Install Python and create a virtual environment: `python -m venv .venv`.
3. Activate it: on Windows PowerShell run `.\.venv\Scripts\Activate.ps1`; on Linux/macOS run `source .venv/bin/activate`.
4. Install exploratory dependencies: `python -m pip install -r requirements-eda.txt`.
5. From the repository root, run `python SCRIPTS/01_explore_raw_data.py --archive "PATH/TO/Names100Dataset.tar.gz" --output OUTPUT/eda`.
6. Inspect the two plots, per-name counts, and `summary.json`. The script also generates a per-image brightness CSV locally. The archive is streamed without changing or extracting original images.

The current results are 80,000 successfully decoded images, 100 name categories, and exactly 800 images per name. Median mean-image brightness is approximately 116.4 on the 0-255 scale.

Full analysis reproduction instructions will be completed after preprocessing and model code are implemented. No training or evaluation results are claimed by this setup.

## References

[3] H. Chen, A. Gallagher, and B. Girod, "What's in a name: First names as facial attributes," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2013, pp. 3366-3373, doi: 10.1109/CVPR.2013.432. [Dataset]. Available: https://purl.stanford.edu/tp945cq9122

[11] J. Guo and J. Deng, "InsightFace: 2D and 3D deep face analysis toolbox," GitHub repository. [Online]. Available: https://github.com/deepinsight/insightface

Reference numbers retain the team's MI2 numbering; consolidate the complete list when the modeling documentation is added.
