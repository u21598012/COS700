from src.tracking.mlflow_logger import start_run
from src.utils.config import load_yaml
from src.evaluation.profiler import get_memory_usage

import mlflow


def test_load_config():
    cfg = load_yaml("configs/experiments.yaml")
    print(cfg)

def test_mlflow_logger():
    with start_run():

        mlflow.log_param("seed", 42)

        mlflow.log_metric("accuracy", 0.9)

def test_memory_usage():
    print(f"Memory usage: {get_memory_usage():.2f} MB")

if __name__ == "__main__":

    # test_load_config()
    # test_mlflow_logger()
    test_memory_usage()