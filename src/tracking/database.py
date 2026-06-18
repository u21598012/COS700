"""SQLite results database (framework spec, Phase 5).

Four tables:
  * experiments  - one row per experiment (metadata + config snapshot)
  * runs         - one row per seed
  * evaluations  - one row per architecture evaluated
  * best_models  - best architecture per run

Initialisation must run before any experiment begins.
"""

import json
import sqlite3
from datetime import datetime, timezone

from src.utils import paths


def _now():
    return datetime.now(timezone.utc).isoformat()


def get_connection(db_path=None):
    """Open a connection, creating the parent directory if needed."""
    db_path = paths.DB_PATH if db_path is None else paths.resolve(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database(db_path=None):
    """Create all tables if they do not yet exist."""
    conn = get_connection(db_path)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            experiment_id   TEXT PRIMARY KEY,
            experiment_name TEXT,
            algorithm       TEXT,
            created_at      TEXT,
            num_seeds       INTEGER,
            budget          INTEGER,
            config_json     TEXT,
            system_info     TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id                 TEXT PRIMARY KEY,
            experiment_id          TEXT,
            algorithm              TEXT,
            seed                   INTEGER,
            mlflow_run_id          TEXT,
            start_time             TEXT,
            end_time               TEXT,
            total_search_time      REAL,
            time_to_best           REAL,
            best_evaluation_number INTEGER,
            best_accuracy          REAL,
            best_precision         REAL,
            best_recall            REAL,
            best_macro_f1          REAL,
            peak_memory_mb         REAL,
            best_architecture      TEXT,
            FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS evaluations (
            evaluation_id     TEXT PRIMARY KEY,
            run_id            TEXT,
            algorithm         TEXT,
            seed              INTEGER,
            evaluation_number INTEGER,
            architecture_json TEXT,
            accuracy          REAL,
            precision_score   REAL,
            recall            REAL,
            macro_f1          REAL,
            training_time     REAL,
            inference_time    REAL,
            avg_cpu           REAL,
            peak_cpu          REAL,
            avg_ram_mb        REAL,
            peak_ram_mb       REAL,
            peak_gpu_mb       REAL,
            peak_memory_mb    REAL,
            failed            INTEGER,
            created_at        TEXT,
            FOREIGN KEY (run_id) REFERENCES runs (run_id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS best_models (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id            TEXT,
            evaluation_number INTEGER,
            architecture_json TEXT,
            accuracy          REAL,
            macro_f1          REAL,
            model_path        TEXT,
            created_at        TEXT,
            FOREIGN KEY (run_id) REFERENCES runs (run_id)
        )
    """)

    conn.commit()
    conn.close()


# --------------------------------------------------------------------------
# Insert / update helpers
# --------------------------------------------------------------------------
def insert_experiment(experiment_id, experiment_name, algorithm, num_seeds,
                      budget, config, system_info, db_path=None):
    conn = get_connection(db_path)
    conn.execute(
        """INSERT OR REPLACE INTO experiments
           (experiment_id, experiment_name, algorithm, created_at, num_seeds,
            budget, config_json, system_info)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (experiment_id, experiment_name, algorithm, _now(), num_seeds, budget,
         json.dumps(config), json.dumps(system_info)),
    )
    conn.commit()
    conn.close()


def create_run(run_id, experiment_id, algorithm, seed, mlflow_run_id=None, db_path=None):
    conn = get_connection(db_path)
    conn.execute(
        """INSERT OR REPLACE INTO runs
           (run_id, experiment_id, algorithm, seed, mlflow_run_id, start_time)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (run_id, experiment_id, algorithm, seed, mlflow_run_id, _now()),
    )
    conn.commit()
    conn.close()


def finalize_run(run_id, summary, db_path=None):
    """Update a run row with its final best-candidate summary."""
    conn = get_connection(db_path)
    conn.execute(
        """UPDATE runs SET
              end_time = ?,
              total_search_time = ?,
              time_to_best = ?,
              best_evaluation_number = ?,
              best_accuracy = ?,
              best_precision = ?,
              best_recall = ?,
              best_macro_f1 = ?,
              peak_memory_mb = ?,
              best_architecture = ?
           WHERE run_id = ?""",
        (
            _now(),
            summary.get("total_search_time"),
            summary.get("time_to_best"),
            summary.get("best_evaluation_number"),
            summary.get("best_accuracy"),
            summary.get("best_precision"),
            summary.get("best_recall"),
            summary.get("best_macro_f1"),
            summary.get("peak_memory_mb"),
            summary.get("best_architecture"),
            run_id,
        ),
    )
    conn.commit()
    conn.close()


def insert_evaluation(evaluation_id, run_id, algorithm, seed, evaluation_number,
                      architecture_json, result, db_path=None):
    """``result`` is an EvaluationResult (or its ``.to_record()`` dict)."""
    rec = result.to_record() if hasattr(result, "to_record") else result
    conn = get_connection(db_path)
    conn.execute(
        """INSERT OR REPLACE INTO evaluations
           (evaluation_id, run_id, algorithm, seed, evaluation_number,
            architecture_json, accuracy, precision_score, recall, macro_f1,
            training_time, inference_time, avg_cpu, peak_cpu, avg_ram_mb,
            peak_ram_mb, peak_gpu_mb, peak_memory_mb, failed, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            evaluation_id, run_id, algorithm, seed, evaluation_number,
            architecture_json,
            rec.get("accuracy"), rec.get("precision"), rec.get("recall"),
            rec.get("macro_f1"), rec.get("training_time"), rec.get("inference_time"),
            rec.get("avg_cpu"), rec.get("peak_cpu"), rec.get("avg_ram_mb"),
            rec.get("peak_ram_mb"), rec.get("peak_gpu_mb"), rec.get("peak_memory_mb"),
            int(bool(rec.get("failed"))), _now(),
        ),
    )
    conn.commit()
    conn.close()


def insert_best_model(run_id, evaluation_number, architecture_json, accuracy,
                      macro_f1, model_path, db_path=None):
    conn = get_connection(db_path)
    conn.execute(
        """INSERT INTO best_models
           (run_id, evaluation_number, architecture_json, accuracy, macro_f1,
            model_path, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (run_id, evaluation_number, architecture_json, accuracy, macro_f1,
         model_path, _now()),
    )
    conn.commit()
    conn.close()


def fetch_all(table, db_path=None):
    """Return every row of ``table`` as a list of dicts (for quick inspection)."""
    if table not in {"experiments", "runs", "evaluations", "best_models"}:
        raise ValueError(f"Unknown table: {table}")
    conn = get_connection(db_path)
    rows = conn.execute(f"SELECT * FROM {table}").fetchall()
    conn.close()
    return [dict(r) for r in rows]
