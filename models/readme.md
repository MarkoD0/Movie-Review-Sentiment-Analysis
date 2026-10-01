# `models/`

Trained models and fitted vectorizers, saved with `.joblib`.
File names follow `<model>_<variant>.joblib`, where the variant is `base`, `stem`, `lem` or `lanc`.

## Included in the repository (~90 MB in total)

| Files | Content |
|---|---|
| `tfidf_vectorizer_<variant>.joblib` | fitted TF-IDF vectorizers (one per preprocessing variant) |
| `naive_bayes_*`, `logistic_regression_*`, `svm_*`, `decision_tree_*` | the four small models (each below 3 MB) |
| `lite_random_forest_*` | compact Random Forests (about 8-10 MB each) |
| `lite_knn_*` | compact KNN models (about 1-1.5 MB each) |
| `svd_reducer_lite_knn_<variant>.joblib` | fitted `TruncatedSVD` reducers required by the lite KNN models |

These files are everything the GUI application (`predict_review.py`) needs.

## NOT included (too large for GitHub)

| File | Size |
|---|---|
| `random_forest_base.joblib` | ~206 MB |
| `random_forest_stem.joblib` | ~6.1 GB |
| `random_forest_lem.joblib` | ~202 MB |
| `random_forest_lanc.joblib` | ~199 MB |
| `knn_base.joblib`, `knn_stem.joblib`, `knn_lem.joblib`, `knn_lanc.joblib` | ~1.5 GB each |

All evaluation metrics of these models are kept in `results/metrics/*.json`.

### How to regenerate the heavy models

Retrain them with the notebooks (after regenerating `data/processed/`, see its README):

| Model | Notebooks | Approximate time on the author's machine (GridSearchCV) |
|---|---|---|
| Random Forest (heavy) | `7.1.0` - `7.1.3` | ~14-15 hours per variant |
| KNN (heavy) | `8.1.0` - `8.1.3` | about 2 hours per variant on average, and a lot of RAM |

The compact "lite" versions were created because these files are impractical to store and share; see `7.2.x` and `8.2.x` and Chapter 4 of the thesis.

## Notes

- Models must be loaded with the same scikit-learn version they were saved with (see `requirements.txt`); other versions may fail to load them or warn about incompatibility.
