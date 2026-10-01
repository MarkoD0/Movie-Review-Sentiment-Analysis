# Loading vectorized data (TF-IDF matrices + labels) saved in 3_vectorization.ipynb
# Used in notebooks 4.x - 9.x (all 12 model notebooks)
# Returns: X_train, X_test, y_train, y_test

from pathlib import Path

import numpy as np
from scipy import sparse


def load_variant(variant: str):
    """
    Returns X_train, X_test, y_train, and y_test for the specified variant.
    """
    data_dir = Path(f"../data/processed/{variant}")

    X_train = sparse.load_npz(data_dir / "X_train.npz")
    X_test = sparse.load_npz(data_dir / "X_test.npz")
    y_train = np.load(data_dir / "y_train.npy")
    y_test = np.load(data_dir / "y_test.npy")

    return X_train, X_test, y_train, y_test
