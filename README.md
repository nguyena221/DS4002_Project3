# Do You Look Like Your Name?

DS4002 Project 3 | 
Authors: Eli Boyden, Zilan Saadi, Annie Nguyen | 
Prof. Loreto Alonzi

## Summary

This project has three main stages: preprocessing, analysis and evaluation. During preprocessing we filter out unidentifiable faces, and faces with an estimated age < 25, and align/crop images. During analysis we fine tune a IResNet-100 model to predict names based on face image data, and produce cosine simularity data from the image embeddings. During evaluation we compare our fine tuned models against a non-fine-tuned IResNet-100 baseline, targeting a statistically significant improvement in simularity lift.

# Preprocessing

This section details steps taken duirng preprocessing for a first-name classifier on the [Names100 dataset](https://purl.stanford.edu/tp945cq9122) (80,000 face images, 100 first names, 800 per name).

## Results

**64,398 images across 100 names**, aligned and split 70/15/15.

| Split | Images |
|---|---:|
| Train | 45,078 |
| Validation | 9,660 |
| Test | 9,660 |

How the 80,000 raw images were reduced:

| Step | Removed |
|---|---:|
| Manually excluded by the group | 69 |
| Not exactly one face (RetinaFace confidence >= 0.5) | 491 |
| Estimated age under 25 (InsightFace `buffalo_l`) | 14,903 |
| Exact duplicate copies | 139 |
| Names with fewer than 300 images left | 0 |

## Layouts

For more information about the data used/cleaned, scripts used, and figures produced, view the individual READMEs linked in the table below.

| Folder | Contents |
|---|---|
| [data/](data/README.md) | Raw archive, exclusion list, and the processed dataset |
| [scripts/](scripts/README.md) | `preprocess.py`, `explore_data.py`, `prepared_dataset.py` |
| [output/](output/README.md) | Exploratory figures |

## Running instructions

Create the preprocessed data from the raw [Names100 dataset](https://purl.stanford.edu/tp945cq9122). This step can take some time depending on the hardware available to you.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\preprocess.py    # needs data/raw/Names100Dataset.tar.gz
```

After preprocessing, you can load the data like this:

```python
import sys
sys.path.insert(0, 'scripts')
from prepared_dataset import PreparedFaces
train = PreparedFaces('data/processed/faces.h5', 'train')
image, label = train[0]   # float32 array (3, 112, 112) in [-1, 1], int label 0-99
```
