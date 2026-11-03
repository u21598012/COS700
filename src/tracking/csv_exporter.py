
import pandas as pd

from src.tracking import database
from src.utils import paths


RUNS_COLUMNS = [
    "algorithm", "seed", "best_accuracy", "best_precision", "best_recall",
    "best_macro_f1", "total_search_time", "time_to_best", "peak_memory_mb",
    "best_architecture",
]

EVALUATIONS_COLUMNS = [
    "algorithm", "seed", "evaluation_number", "architecture_json", "accuracy",
    "precision", "recall", "macro_f1", "training_time", "peak_memory_mb",
]


def export_runs_csv(out_path=None, db_path=None):
    out_path = paths.RUNS_CSV if out_path is None else paths.resolve(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(database.fetch_all("runs", db_path=db_path))
    df = _select(df, RUNS_COLUMNS)
    df.to_csv(out_path, index=False)
    return out_path


def export_evaluations_csv(out_path=None, db_path=None):
    out_path = paths.EVALUATIONS_CSV if out_path is None else paths.resolve(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(database.fetch_all("evaluations", db_path=db_path))
    if "precision_score" in df.columns:
        df = df.rename(columns={"precision_score": "precision"})
    df = _select(df, EVALUATIONS_COLUMNS)
    df.to_csv(out_path, index=False)
    return out_path


def export_all(db_path=None):
    return {
        "runs_csv": export_runs_csv(db_path=db_path),
        "evaluations_csv": export_evaluations_csv(db_path=db_path),
    }


def _select(df, columns):
    for col in columns:
        if col not in df.columns:
            df[col] = None
    return df[columns]
