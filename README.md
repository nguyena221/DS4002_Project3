# Do You Look Like Your Name?

**DS4002 Project 3**
**Group:** EAZ Inc.
**Authors:** Eli Boyden, Zilan Saadi, Annie Nguyen  
**Instructor:** Prof. Loreto Alonzi

## Project Summary

This project investigates whether facial images contain information associated with a person's first name. The project consists of three main stages: preprocessing, model training, and evaluation.

During preprocessing, we filter out images without identifiable faces and faces with an estimated age below 25, then align and crop the remaining faces. During model training, we fine-tune an IResNet-100 model using three metric-learning loss functions: Triplet Margin, ArcFace, and SupCon. We then compare the resulting image embeddings against those produced by a non-fine-tuned IResNet-100 baseline. Finally, we evaluate the models using triplet accuracy, similarity lift, and paired bootstrap confidence intervals.

## Software and Versions

The following dependencies and versions were used:

```text
albucore==0.0.24
albumentations==2.0.8
annotated-types==0.8.0
certifi==2026.10.23
charset-normalizer==3.5.2
cloudpickle==3.1.2
colorama==0.4.6
contourpy==1.3.3
cycler==0.12.1
Cython==3.3.1
easydict==1.14
flatbuffers==25.12.19
fonttools==4.68.0
h5py==3.16.0
idna==3.20
ImageIO==2.38.1
insightface==0.7.3
joblib==1.6.0
kiwisolver==1.5.1
lazy-loader==0.6
matplotlib==3.11.2
ml_dtypes==0.5.4
narwhals==2.28.0
networkx==3.6.1
numpy==1.26.4
onnx==1.23.2
onnxruntime==1.32.0
opencv-python-headless==4.11.0.86
packaging==26.3
pillow==13.0.0
prettytable==3.18.0
protobuf==7.36.2
pytorch-metric-learning==2.8.1
pydantic==2.14.3
pydantic_core==2.50.4
pyparsing==3.3.3
python-dateutil==4.0.0
PyYAML==6.0.3
requests==2.34.2
scikit-image==0.26.0
scikit-learn==1.9.1
scipy==1.17.1
simsimd==6.5.16
stringzilla==5.3.2
threadpoolctl==3.8.0
torch==2.8.0
tifffile==2026.3.3
tqdm==4.70.1
typing-inspection==0.4.5
typing_extensions==4.16.0
tzdata==2026.6
uniface==4.3.0
urllib3==2.8.0
wcwidth==0.9.4
```

## Execution Environments

- **Preprocessing:** macOS, Python 3.11
- **Model training:** LooseLeaf Linux environment, Python 3.10.12, NVIDIA L4 GPU, CUDA 12.8, and PyTorch 2.8.0+cu128
- **Final evaluation:** Google Colab

## Repository Structure

```text
DS4002_Project3/
├── README.md
├── LICENSE.md
├── requirements.txt
├── data/
│   ├── README.md
│   ├── metadata.md (contains Data Set Establishment and Details from MI2)
│   ├── exclusions.csv
│   ├── raw/ (Names100Dataset.tar.gz excluded from git)
│   └── processed/
│       ├── manifest.csv
│       └── faces.h5 (excluded from git)
├── scripts/
│   ├── preprocess/
│   │   ├── README.md
│   │   ├── explore_data.py
│   │   ├── preprocess.py
│   │   └── prepared_dataset.py
│   ├── training/
│   │   ├── README.md
│   │   ├── face_name_metric_learning/
│   │   │   ├── README.md
│   │   │   ├── __init__.py
│   │   │   ├── acceptance.py
│   │   │   ├── audit.py
│   │   │   ├── handoff.py
│   │   │   ├── iresnet.py
│   │   │   ├── pytorch_training.py
│   │   │   └── training.py
│   │   ├── face_name_metrics/
│   │   │   ├── README.md
│   │   │   ├── __init__.py
│   │   │   ├── bootstrap.py
│   │   │   └── metrics.py
│   │   └── notebooks/
│   │       ├── README.md
│   │       ├── setup.ipynb
│   │       ├── 00_setup_and_smoke_test.ipynb
│   │       ├── 01_export_baseline_embeddings.ipynb
│   │       ├── 02_train_metric_models.ipynb
│   │       └── 03_export_selected_embeddings.ipynb
│   └── eval/
│       ├── README.md
│       └── DS_4002_Project_3_Evaluation.ipynb
└── output/
    ├── README.md
    ├── final_metrics.csv
    ├── artifacts/
    │   ├── README.md
    │   ├── handoff_manifest.json
    │   ├── setup_environment.json
    │   ├── trial_results.csv
    │   └── embeddings/
    │       ├── baseline_iresnet100.npz
    │       ├── triplet_margin_selected.npz
    │       ├── arcface_selected.npz
    │       └── supcon_selected.npz
    └── eda/
        ├── README.md
        ├── 01_images_per_name.png
        ├── 02_image_brightness.png
        ├── name_counts.csv
        └── summary.json
```

For additional details about each component, see the corresponding documentation:

- [`data/README.md`](data/README.md): Dataset sources, contents, and data organization.
- [`scripts/preprocess/README.md`](scripts/preprocess/README.md): Preprocessing scripts and procedures.
- [`scripts/training/README.md`](scripts/training/README.md): Training pipeline overview.
  - [`face_name_metric_learning/README.md`](scripts/training/face_name_metric_learning/README.md): Model training, packaging, and auditing utilities.
  - [`face_name_metrics/README.md`](scripts/training/face_name_metrics/README.md): Triplet accuracy and statistical evaluation metrics.
  - [`notebooks/README.md`](scripts/training/notebooks/README.md): Training notebook execution order and required inputs.
- [`scripts/eval/README.md`](scripts/eval/README.md): Final evaluation procedure.
- [`output/README.md`](output/README.md): Generated results and artifacts.
  - [`artifacts/README.md`](output/artifacts/README.md): Training outputs and embedding files.
  - [`eda/README.md`](output/eda/README.md): Exploratory data analysis outputs.

## Reproducing Results

Follow the steps below in order to reproduce the project's preprocessing, model training, and evaluation results.

### 1. Obtain the Dataset

Download the original Name100 dataset from the [Stanford Digital Repository](https://purl.stanford.edu/tp945cq9122) and place the archive at:

```text
data/raw/Names100Dataset.tar.gz
```

The `data/exclusions.csv` file contains the manually excluded images and their corresponding SHA-256 hashes and exclusion reasons.

See [`data/README.md`](data/README.md) for additional information about the dataset and its organization.

### 2. Install Dependencies

From the repository root, install the dependencies:

```bash
pip install -r requirements.txt
```

The preprocessing environment uses Python 3.11. Model training was performed separately on LooseLeaf using Python 3.10.12 and a CUDA-enabled GPU environment.

### 3. Explore and Preprocess the Data

Use the exploratory data analysis script, [`explore_data.py`](/scripts/preprocess/explore_data.py) to produce initial EDA on the raw dataset.

This generates the exploratory figures and summary statistics in `output/eda/`.

Next, run the preprocessing pipeline found in [`preprocess.py`](/scripts/preprocess/preprocess.py):

The pipeline detects faces, estimates age, aligns and crops images, removes duplicates, filters classes, and creates stratified training, validation, and test splits. It produces the data:

- `data/processed/faces.h5` (not committed)
- `data/processed/manifest.csv`

The resulting dataset contains 64,398 aligned face images across 100 name classes, split into 45,078 training images, 9,660 validation images, and 9,660 test images.

See [`scripts/preprocess/README.md`](scripts/preprocess/README.md) for the preprocessing details and requirements.

### 4. Train the Models

Use the training notebooks in `scripts/training/notebooks/` on [LooseLeaf](https://looseleafcloud.org/). Run them in the following order:

1. `setup.ipynb` – Set up the pretrained IResNet-100 weights.
2. `00_setup_and_smoke_test.ipynb` – Verify the environment, dataset, pretrained weights, and model execution.
3. `01_export_baseline_embeddings.ipynb` – Generate embeddings using the pretrained, non-fine-tuned IResNet-100 baseline.
4. `02_train_metric_models.ipynb` – Run the 10-trial training process across Triplet Margin, ArcFace, and SupCon loss functions and select the best model for each.
5. `03_export_selected_embeddings.ipynb` – Export the selected models' validation and test embeddings and generate the handoff manifest.

The training process requires the preprocessed dataset, pretrained IResNet-100 weights, and a CUDA-enabled GPU environment.

See [`scripts/training/notebooks/README.md`](scripts/training/notebooks/README.md) for notebook-specific instructions and requirements.

### 5. Evaluate the Models

Run [`scripts/eval/DS_4002_Project_3_Evaluation.ipynb`](/scripts/eval/DS_4002_Project_3_Evaluation.ipynb) in Google Colab. The evaluation notebook validates the training handoff artifacts, calculates triplet accuracy for the baseline and three fine-tuned models, and performs 2,000 paired bootstrap replicates.

The evaluation produces `output/final_metrics.csv`, containing the final triplet accuracy, similarity lift over the baseline, and 95% confidence intervals for each model on the validation and test splits.

See [`scripts/eval/README.md`](scripts/eval/README.md) for the complete evaluation workflow and required inputs.

## Results

The final evaluation results are available in [`output/final_metrics.csv`](output/final_metrics.csv). Additional training artifacts and exploratory data analysis outputs are documented in [`output/README.md`](output/README.md).