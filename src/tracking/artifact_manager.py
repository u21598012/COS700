

import json

import numpy as np
import pandas as pd
import torch

from src.utils import paths


def initialize_artifacts():
    paths.ensure_dirs()


# --------------------------------------------------------------------------
def save_architecture(architecture, name, directory=None):
    directory = paths.ARCHITECTURES_DIR if directory is None else paths.resolve(directory)
    directory.mkdir(parents=True, exist_ok=True)
    filepath = directory / f"{name}.json"
    payload = architecture.to_dict() if hasattr(architecture, "to_dict") else architecture
    with open(filepath, "w") as f:
        json.dump(payload, f, indent=4)
    return filepath


def save_model(model, name, directory=None):
    directory = paths.MODELS_DIR if directory is None else paths.resolve(directory)
    directory.mkdir(parents=True, exist_ok=True)
    filepath = directory / f"{name}.pt"
    torch.save(model.state_dict(), filepath)
    return filepath


def save_predictions(y_true, y_pred, name, directory=None):
    directory = paths.PREDICTIONS_DIR if directory is None else paths.resolve(directory)
    directory.mkdir(parents=True, exist_ok=True)
    filepath = directory / f"{name}.csv"
    pd.DataFrame({"y_true": np.asarray(y_true), "y_pred": np.asarray(y_pred)}).to_csv(
        filepath, index=False
    )
    return filepath


def save_search_progress(rows, name, directory=None):
    directory = paths.LOGS_DIR if directory is None else paths.resolve(directory)
    directory.mkdir(parents=True, exist_ok=True)
    filepath = directory / f"{name}.csv"
    pd.DataFrame(rows).to_csv(filepath, index=False)
    return filepath


def save_confusion_matrix(cm, name, class_names=None, directory=None):
    directory = paths.PLOTS_DIR if directory is None else paths.resolve(directory)
    directory.mkdir(parents=True, exist_ok=True)
    filepath = directory / f"{name}.png"
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues")
        fig.colorbar(im, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(name)
        if class_names is not None:
            ax.set_xticks(range(len(class_names)))
            ax.set_yticks(range(len(class_names)))
            ax.set_xticklabels(class_names)
            ax.set_yticklabels(class_names)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8)
        fig.tight_layout()
        fig.savefig(filepath, dpi=120)
        plt.close(fig)
        return filepath
    except Exception:
        return None
