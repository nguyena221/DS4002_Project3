# Exploratory plots for MI2

**Figure 1. Raw image counts by first name.** The Names100 dataset [3] contains 80,000 images distributed evenly across 100 first-name categories, with exactly 800 images per name. Counts were calculated directly from the downloaded archive. This answers the exploratory question, "Are first-name categories equally represented in the raw dataset?" Adult and image-quality screening may change this balance, so category counts will be reassessed after preprocessing.

**Figure 2. Distribution of mean image brightness.** Mean brightness was calculated for each of the 80,000 images in the Names100 dataset [3] by averaging the weighted RGB value (0.299R + 0.587G + 0.114B) across all pixels. Values range from 0 (black) to 255 (white). The distribution is approximately unimodal and centered near 116.4, with 90% of images between 62.5 and 169.8. The observed spread describes variation in overall image brightness, which may reflect lighting, backgrounds, and image content. Brightness alone does not establish image quality or provide a basis for automatic exclusion.

**Current unknown and planned mitigation:** The number of usable images judged to depict adults in each first-name category remains unknown. Adult eligibility and image-quality screening will be documented before splitting the data, and category counts will be recalculated afterward.

## Reproducibility

- Both figures use the complete raw archive, before screening or train/validation/test splitting.
- All 80,000 images decoded successfully. This does not verify label correctness, age, face detection, or unique identities.
- `name_counts.csv` contains category counts; `image_brightness.csv` contains one measurement per image; `summary.json` contains descriptive statistics.
- `../../scripts/explore_data.py` generates the figures and measurements. Requires Python with numpy, Pillow, and matplotlib.
- Example: `python scripts/explore_data.py --archive "C:\Users\annie\Downloads\Names100Dataset.tar.gz" --output output/eda`
- [3] refers to the Chen, Gallagher, and Girod dataset reference already in the MI2 reference list. The plots and calculated statistics are the group's own exploratory work.
