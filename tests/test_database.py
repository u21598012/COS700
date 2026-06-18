"""Database init + CRUD round-trip tests (uses a temp DB, not the real one)."""

import sys
import tempfile
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.evaluator import EvaluationResult
from src.tracking import database


def _temp_db():
    return Path(tempfile.mkdtemp()) / "test_results.db"


def test_initialize_creates_tables():
    db = _temp_db()
    database.initialize_database(db)
    for table in ("experiments", "runs", "evaluations", "best_models"):
        assert database.fetch_all(table, db_path=db) == []


def test_full_round_trip():
    db = _temp_db()
    database.initialize_database(db)

    exp_id, run_id = uuid.uuid4().hex, uuid.uuid4().hex
    database.insert_experiment(exp_id, "TEST", "random_search", 1, 2, {"a": 1}, {"os": "test"}, db_path=db)
    database.create_run(run_id, exp_id, "random_search", 11, db_path=db)

    result = EvaluationResult(accuracy=0.9, precision=0.8, recall=0.7, macro_f1=0.75,
                              training_time=1.2, peak_memory_mb=100.0)
    database.insert_evaluation(uuid.uuid4().hex, run_id, "random_search", 11, 1,
                               '{"learning_rate": 0.001, "num_layers": 1}', result, db_path=db)
    database.finalize_run(run_id, {"total_search_time": 5.0, "best_accuracy": 0.9,
                                   "best_macro_f1": 0.75, "best_architecture": "{}"}, db_path=db)
    database.insert_best_model(run_id, 1, "{}", 0.9, 0.75, "results/models/x.pt", db_path=db)

    assert len(database.fetch_all("experiments", db_path=db)) == 1
    assert len(database.fetch_all("evaluations", db_path=db)) == 1
    assert len(database.fetch_all("best_models", db_path=db)) == 1
    run = database.fetch_all("runs", db_path=db)[0]
    assert run["best_accuracy"] == 0.9
    assert run["total_search_time"] == 5.0
    ev = database.fetch_all("evaluations", db_path=db)[0]
    assert ev["precision_score"] == 0.8  # stored under precision_score (SQL keyword)


if __name__ == "__main__":
    test_initialize_creates_tables()
    test_full_round_trip()
    print("test_database: OK")
