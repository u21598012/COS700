
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.utils import paths


def splits_exist():
    return (
        paths.TRAIN_IDX_PATH.exists()
        and paths.VAL_IDX_PATH.exists()
        and paths.TEST_IDX_PATH.exists()
    )


def require_splits(verbose=True):

    if not splits_exist():
        raise FileNotFoundError(
            "Fixed splits not found in data/splits/. The framework does not create "
            "splits - create them once yourself and save:\n"
            f"    {paths.TRAIN_IDX_PATH}\n"
            f"    {paths.VAL_IDX_PATH}\n"
            f"    {paths.TEST_IDX_PATH}\n"
            "(train_idx/val_idx index mitbih_train.csv rows; test_idx indexes "
            "mitbih_test.csv rows.)"
        )
    if verbose:
        print(f"[splits] Loaded existing fixed splits from {paths.SPLITS_DIR}.")
    return load_splits()


def load_splits():
    for p in (paths.TRAIN_IDX_PATH, paths.VAL_IDX_PATH, paths.TEST_IDX_PATH):
        if not p.exists():
            raise FileNotFoundError(
                f"Split file missing: {p}. The framework only loads splits - "
                "create the .npy index files once yourself first."
            )
    return (
        np.load(paths.TRAIN_IDX_PATH),
        np.load(paths.VAL_IDX_PATH),
        np.load(paths.TEST_IDX_PATH),
    )


def _subsample(X, y, n, seed=0):
    if not n or n >= len(X):
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=n, replace=False)
    return X[idx], y[idx]


def _make_loader(X, y, batch_size, shuffle, num_workers, pin_memory):
    dataset = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def _split_xy(arr):
    X = arr[:, :-1].astype(np.float32)
    y = arr[:, -1].astype(np.int64)
    return X, y


def get_dataloaders(dataset_cfg, batch_size=None, max_train_samples=None,
                    max_val_samples=None, max_test_samples=None, nrows=None):
   
    dl_cfg = dataset_cfg.get("dataloader", {})
    pre_cfg = dataset_cfg.get("preprocessing", {})

    batch_size = batch_size or dl_cfg.get("batch_size", 128)
   
    num_workers = int(pre_cfg.get("num_workers", 0))
    pin_memory = dl_cfg.get("pin_memory", False) and torch.cuda.is_available()
    shuffle_train = pre_cfg.get("shuffle_train", True)

    train_arr, val_arr, test_arr = load_splits()
    X_train, y_train = _split_xy(train_arr)
    X_val, y_val = _split_xy(val_arr)
    X_test, y_test = _split_xy(test_arr)

    X_train, y_train = _subsample(X_train, y_train, max_train_samples)
    X_val, y_val = _subsample(X_val, y_val, max_val_samples)
    X_test, y_test = _subsample(X_test, y_test, max_test_samples)

    loaders = {
        "train": _make_loader(X_train, y_train, batch_size, shuffle_train, num_workers, pin_memory),
        "val": _make_loader(X_val, y_val, batch_size, False, num_workers, pin_memory),
        "test": _make_loader(X_test, y_test, batch_size, False, num_workers, pin_memory),
    }
    num_classes = dataset_cfg.get("dataset", {}).get("num_classes")
    meta = {
        "input_dim": X_train.shape[1],
        "num_classes": int(num_classes) if num_classes else int(np.unique(y_train).size),
    }
    return loaders, meta
