"""Central definition of every path the framework relies on.

The framework assumes these directories exist (see framework spec, Phase 2 / 4).
Import ``PATHS`` and call :func:`ensure_dirs` once at startup to create them.
"""

from pathlib import Path


# Project root = two levels up from this file (src/utils/paths.py -> project root).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- configs ---------------------------------------------------------------
CONFIG_DIR = PROJECT_ROOT / "configs"

# --- data ------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"

TRAIN_IDX_PATH = SPLITS_DIR / "train_idx.npy"
VAL_IDX_PATH = SPLITS_DIR / "val_idx.npy"
TEST_IDX_PATH = SPLITS_DIR / "test_idx.npy"

# --- results ---------------------------------------------------------------
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

# --- mlflow ----------------------------------------------------------------
MLRUNS_DIR = PROJECT_ROOT / "mlruns"


# Directories that must always exist before an experiment runs.
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
    """Create every results/data directory the framework writes to."""
    for directory in _DIRS_TO_CREATE:
        directory.mkdir(parents=True, exist_ok=True)


def resolve(path):
    """Resolve ``path`` relative to the project root if it is not absolute."""
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p)
