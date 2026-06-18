"""Raw MIT-BIH CSV loading.

The MIT-BIH arrhythmia CSVs have no header: the first 187 columns are the ECG
signal samples and the final column is the integer class label (0-4).
"""

import numpy as np
import pandas as pd

from src.utils import paths

# Number of signal samples per heartbeat in the MIT-BIH CSVs.
SIGNAL_LENGTH = 187
NUM_CLASSES = 5


def _csv_path(dataset_cfg, which):
    root = paths.resolve(dataset_cfg["dataset"]["root_dir"])
    filename = dataset_cfg["dataset"][f"{which}_file"]
    return root / filename


def load_raw(dataset_cfg, which="train", nrows=None):
    """Load one split CSV into ``(X, y)`` numpy arrays.

    Parameters
    ----------
    dataset_cfg : dict
        Parsed ``dataset.yaml``.
    which : {"train", "test"}
        Which CSV to load.
    nrows : int, optional
        Limit the number of rows read (handy for fast smoke runs).

    Returns
    -------
    X : np.ndarray, shape (N, 187), float32
    y : np.ndarray, shape (N,), int64
    """
    path = _csv_path(dataset_cfg, which)
    df = pd.read_csv(path, header=None, nrows=nrows)
    values = df.values
    X = values[:, :SIGNAL_LENGTH].astype(np.float32)
    y = values[:, -1].astype(np.int64)
    return X, y


def load_labels(dataset_cfg, which="train"):
    """Load only the label column (cheap - used for stratified splitting)."""
    path = _csv_path(dataset_cfg, which)
    labels = pd.read_csv(path, header=None, usecols=[SIGNAL_LENGTH]).values.ravel()
    return labels.astype(np.int64)
