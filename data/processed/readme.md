# `data/processed/`

The processed data (~370 MB in total) is **not** stored in this repository because it is fully regenerable from the raw dataset.

## What goes here

One sub-folder per text preprocessing variant:

| Folder | Variant |
|---|---|
| `baseline/` | cleaned text, no normalization |
| `stemmed/` | Snowball (Porter2) stemming |
| `lemmatized/` | WordNet lemmatization with POS tagging |
| `lancaster/` | Lancaster (aggressive) stemming |

Each folder contains:

| File | Created by | Content |
|---|---|---|
| `train.csv`, `test.csv` | `notebooks/2_preprocessing.ipynb` | processed review text + `sentiment` label (39,665 train / 9,917 test rows) |
| `X_train.npz`, `X_test.npz` | `notebooks/3_vectorization.ipynb` | sparse TF-IDF matrices (unigrams + bigrams) |
| `y_train.npy`, `y_test.npy` | `notebooks/3_vectorization.ipynb` | binary labels (`positive` = 1, `negative` = 0) |

## How to regenerate

1. Put the raw dataset in `data/raw/` (see `data/raw/README.md`).
2. From the `notebooks/` directory, run in order:
   1. `2_preprocessing.ipynb` (on the author's machine the lemmatization step alone took about 5 minutes, and the two stemming steps about 1 minute each)
   2. `3_vectorization.ipynb`

The split is stratified 80/20 with `random_state=1` and is done once, before the four variants branch, so all variants share the same train and test reviews.

The vectorizers are fitted on the training set only.

Pin the library versions from `requirements.txt` to stay as close as possible to the original run.

The fitted TF-IDF vectorizers are saved to `models/` (and are included in the repository).
