

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_DIR = PROJECT_ROOT / "configs"

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"

TRAIN_IDX_PATH = SPLITS_DIR / "train_idx.npy"
VAL_IDX_PATH = SPLITS_DIR / "val_idx.npy"
TEST_IDX_PATH = SPLITS_DIR / "test_idx.npy"

RESULTS_DIR = PROJECT_ROOT / "results"
DATABASE_DIR = RESULTS_DIR / "database"
CSV_DIR = RESULTS_DIR / "csv"
MODELS_DIR = RESULTS_DIR / "models"
ARCHITECTURES_DIR = RESULTS_DIR / "architectures"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
PLOTS_DIR = RESULTS_DIR / "plots"
LOGS_DIR = RESULTS_DIR / "logs"

DB_PATH = DATABASE_DIR / "results.db"
RUNS_CSV = CSV_DIR / "runs.csv"
EVALUATIONS_CSV = CSV_DIR / "evaluations.csv"

SURROGATE_DATASET_CSV = CSV_DIR / "surrogate_dataset.csv"

MLRUNS_DIR = PROJECT_ROOT / "mlruns"


_DIRS_TO_CREATE = [
    PROCESSED_DIR,
    SPLITS_DIR,
    RESULTS_DIR,
    DATABASE_DIR,
    CSV_DIR,
    MODELS_DIR,
    ARCHITECTURES_DIR,
    PREDICTIONS_DIR,
    PLOTS_DIR,
    LOGS_DIR,
]


def ensure_dirs():
    for directory in _DIRS_TO_CREATE:
        directory.mkdir(parents=True, exist_ok=True)


def surrogate_dataset_path(run_label):
   
    return CSV_DIR / "surrogate_datasets" / f"{run_label}.csv"


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p)
