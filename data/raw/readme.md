# `data/raw/`

The raw dataset is **not** stored in this repository (it is third-party data; see the licence and terms on its Kaggle page).

## What goes here

| File | Description |
|---|---|
| `IMDB_Dataset.csv` | IMDB Dataset of 50K Movie Reviews (columns: `review`, `sentiment`) |

## How to obtain it

1. Download the dataset from Kaggle:
   <https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews>
2. Place the CSV in this folder and name it exactly `IMDB_Dataset.csv`
   (the file downloaded from Kaggle may have a different name, e.g. with a space instead of the underscore; the notebooks expect `../data/raw/IMDB_Dataset.csv`).

Or with the Kaggle CLI:

```bash
pip install kaggle
kaggle datasets download -d lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
```

## Dataset facts (as observed in `notebooks/1_eda.ipynb`)

- 50,000 reviews, perfectly balanced: 25,000 positive and 25,000 negative
- No missing values
- 418 exact duplicate rows (removed in `2_preprocessing.ipynb`, leaving 49,582 reviews)
- 29,200 reviews contain HTML tags (mostly `<br />`)

## Source

The Kaggle page points to the Stanford IMDB sentiment page as the original source of the data:
<http://ai.stanford.edu/~amaas/data/sentiment/>

Original publication:
A. L. Maas, R. E. Daly, P. T. Pham, D. Huang, A. Y. Ng, C. Potts,
*Learning Word Vectors for Sentiment Analysis*, ACL 2011.
