"""MLflow tracking wrapper (framework spec, Phase 6).

Each seed becomes a separate MLflow run. The tracking URI / experiment name come
from config (``experiment.yaml`` -> ``tracking_uri`` / ``experiment_name``, with
``tracking.yaml`` -> ``mlflow`` as a fallback).
"""

import mlflow


def setup_mlflow(tracking_uri, experiment_name):
    """Point MLflow at the tracking store and select the experiment."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return mlflow.get_tracking_uri()


def resolve_mlflow_settings(configs):
    """Pick tracking_uri / experiment_name from the loaded config dict.

    Prefers ``experiment.yaml`` (what the user actually edits), falls back to
    ``tracking.yaml``'s ``mlflow`` section.
    """
    experiment = configs.get("experiment", {})
    mlflow_cfg = configs.get("tracking", {}).get("mlflow", {})

    tracking_uri = experiment.get("tracking_uri") or mlflow_cfg.get("tracking_uri")
    experiment_name = experiment.get("experiment_name") or mlflow_cfg.get("experiment_name")
    if not experiment_name:
        raise ValueError("experiment_name missing from experiment.yaml and tracking.yaml")
    return tracking_uri, experiment_name


def start_run(run_name=None, nested=False, tags=None):
    """Start (and return) a new MLflow run. Use as a context manager."""
    return mlflow.start_run(run_name=run_name, nested=nested, tags=tags)


def log_params(params):
    mlflow.log_params(params)


def log_metrics(metrics, step=None):
    # MLflow only accepts numeric metrics; silently skip the rest.
    numeric = {k: v for k, v in metrics.items() if isinstance(v, (int, float)) and v is not None}
    mlflow.log_metrics(numeric, step=step)


def log_artifact(local_path, artifact_path=None):
    mlflow.log_artifact(str(local_path), artifact_path=artifact_path)
