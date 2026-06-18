"""MLflow settings resolution + a real (file-backed) logged run."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tracking import mlflow_logger


def test_resolve_prefers_experiment_yaml():
    configs = {
        "experiment": {"tracking_uri": "sqlite:///mlflow.db", "experiment_name": "MITBIH_RS"},
        "tracking": {"mlflow": {"tracking_uri": "file:./mlruns", "experiment_name": "FALLBACK"}},
    }
    uri, name = mlflow_logger.resolve_mlflow_settings(configs)
    assert uri == "sqlite:///mlflow.db"
    assert name == "MITBIH_RS"


def test_resolve_falls_back_to_tracking_yaml():
    configs = {"experiment": {}, "tracking": {"mlflow": {"experiment_name": "FALLBACK"}}}
    _, name = mlflow_logger.resolve_mlflow_settings(configs)
    assert name == "FALLBACK"


def test_log_a_run():
    import mlflow

    tmp = Path(tempfile.mkdtemp())
    # Use a sqlite backend (the file store is deprecated in modern MLflow), which
    # is also what the framework uses by default.
    mlflow_logger.setup_mlflow(f"sqlite:///{(tmp / 'mlflow.db').as_posix()}", "unit_test")
    with mlflow_logger.start_run(run_name="t"):
        mlflow_logger.log_params({"seed": 11})
        mlflow_logger.log_metrics({"accuracy": 0.9, "note": "ignored-nonnumeric"})
    runs = mlflow.search_runs(experiment_names=["unit_test"])
    assert len(runs) >= 1


if __name__ == "__main__":
    test_resolve_prefers_experiment_yaml()
    test_resolve_falls_back_to_tracking_yaml()
    test_log_a_run()
    print("test_mlflow: OK")
