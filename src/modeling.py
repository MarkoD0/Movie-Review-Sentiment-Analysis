# Shared function for training, evaluating, and saving model results train_and_evaluate()
#
# Used in notebooks 4.x - 9.x (all 12 model notebooks)
#
# Steps:
#   1. GridSearchCV on the TRAIN set:
#       - tests all hyperparameter combinations from param_grid
#       - uses k-fold cross-validation (5 folds by default)
#       - selects the hyperparameter combination that achieves the best F1 score
#       - the TEST set is not used during this process
#   2. Final model:
#       - after selecting the best hyperparameters, GridSearchCV trains the best model on the entire TRAIN set
#   3. Evaluation:
#       - the best model is used on the TEST set for the first time
#       - accuracy, precision, recall, and F1 are calculated
#       - a confusion matrix is generated
#   4. Saving results:
#       - models/            -> trained best models (.joblib)
#       - results/metrics/   -> metrics and best hyperparameters (.json)
#       - results/figures/   -> confusion matrix images (.png)
#
#
# Function parameters:
#   - model             - sklearn model being trained
#   - param_grid        - dictionary of hyperparameters tested by GridSearchCV
#   - X_train, y_train  - data used for training and hyperparameter selection
#   - X_test, y_test    - separate data used for final evaluation
#   - model_name        - model name
#   - variant           - preprocessing variant (stemming/lemmatization)
#   - cv_folds          - number of folds for GridSearchCV (5 by default)
#
# The function returns a dictionary containing metrics and the best hyperparameters
#
#
# Metrics:
#   - accuracy          - percentage of all correct predictions
#   - precision         - of all predicted positive examples, how many are actually positive
#   - recall            - of all actual positive examples, how many were found by the model
#   - f1                - combines precision and recall into a single metric, F1 = 2 * (precision * recall) / (precision + recall)
#   - confusion matrix  - counts for: True Negative, False Positive, False Negative, True Positive


import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix)


def train_and_evaluate(model, param_grid: dict, X_train, y_train, X_test, y_test, model_name: str, variant: str, cv_folds: int = 5) -> dict:
    """
    Trains the model with GridSearchCV and selects the best hyperparameters using k-fold cross-validation on the training set.
    Evaluates the best model on the test set, saves the model, metrics, and confusion matrix, and returns the results.
    """
    print(f"========== {model_name} ({variant}) ==========")

    # ---------------------------------------------------------------------------
    # Training the model on the train set
    # ---------------------------------------------------------------------------
    #
    # GridSearchCV - hyperparameters are selected exclusively on the train set (through inner k-fold cross-validation)
    # The best hyperparameter combination is selected based on F1 score.
    # The model is trained on the entire train set using the best hyperparameters.
    # The test set remains untouched until the final evaluation
    start = time.time()
    grid_search = GridSearchCV(model, param_grid, cv=cv_folds, scoring="f1", n_jobs=-1)     # n_jobs=-1 use all availabe CPU cores
    grid_search.fit(X_train, y_train)
    train_time = time.time() - start

    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"GridSearchCV time: {train_time:.1f}s")


    # ---------------------------------------------------------------------------
    # Testing the model on the test set (the model sees the test set for the first time here)
    # ---------------------------------------------------------------------------
    y_pred = best_model.predict(X_test)

    metrics = {
        "model_name": model_name,
        "variant": variant,
        "best_params": grid_search.best_params_,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "train_time_seconds": train_time,
    }

    print(f"\nAccuracy  : {metrics['accuracy']:.4f}")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1        : {metrics['f1']:.4f}")

    # Confusion matrix (Provides counts for: True Negative, False Positive, False Negative, True Positive)
    cm = confusion_matrix(y_test, y_pred)
    cm_percent = cm / cm.sum(axis=1, keepdims=True) * 100
    annotations = [[f"{count}\n({percent:.1f}%)" for count, percent in zip(row_count, row_percent)]for row_count, row_percent in zip(cm, cm_percent)]
    fig, ax = plt.subplots(figsize=(5, 4))
    sn.heatmap(cm, annot=annotations, fmt="", cmap="Blues", xticklabels=["negative", "positive"], yticklabels=["negative", "positive"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{model_name} ({variant})")
    plt.tight_layout()

    fig_path = Path("../results/figures") / f"{model_name}_{variant}_confusion_matrix.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    plt.show()


    # ---------------------------------------------------------------------------
    # Saving data
    # ---------------------------------------------------------------------------
    # Saving the model
    model_path = Path("../models") / f"{model_name}_{variant}.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_path)

    # Saving metrics
    metrics_path = Path("../results/metrics") / f"{model_name}_{variant}.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved:")
    print(f"  {model_path}")
    print(f"  {metrics_path}")
    print(f"  {fig_path}")

    return metrics
