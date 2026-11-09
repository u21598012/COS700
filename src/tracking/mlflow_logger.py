

import mlflow


def setup_mlflow(tracking_uri, experiment_name):
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return mlflow.get_tracking_uri()


def resolve_mlflow_settings(configs):
  
    experiment = configs.get("experiment", {})
    mlflow_cfg = configs.get("tracking", {}).get("mlflow", {})

    tracking_uri = experiment.get("tracking_uri") or mlflow_cfg.get("tracking_uri")
    experiment_name = experiment.get("experiment_name") or mlflow_cfg.get("experiment_name")
    if not experiment_name:
        raise ValueError("experiment_name missing from experiment.yaml and tracking.yaml")
    return tracking_uri, experiment_name


def start_run(run_name=None, nested=False, tags=None):
    return mlflow.start_run(run_name=run_name, nested=nested, tags=tags)


def log_params(params):
    mlflow.log_params(params)


def log_metrics(metrics, step=None):
    numeric = {k: v for k, v in metrics.items() if isinstance(v, (int, float)) and v is not None}
    mlflow.log_metrics(numeric, step=step)


def log_artifact(local_path, artifact_path=None):
    mlflow.log_artifact(str(local_path), artifact_path=artifact_path)
