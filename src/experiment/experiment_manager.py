
import uuid

from src.algorithms import ALGORITHMS
from src.data import split_loader
from src.experiment.runner import Runner
from src.models.search_space import SearchSpace
from src.tracking import artifact_manager, csv_exporter, database, mlflow_logger
from src.utils import paths
from src.utils.config import load_configs
from src.utils.system_info import get_device, get_system_info


class ExperimentManager:
    def __init__(self, configs=None, device="auto", db_path=None):
        self.configs = configs if configs is not None else load_configs()
        self.experiment_cfg = self.configs["experiment"]
        self.dataset_cfg = self.configs["dataset"]
        self.device = get_device(device)
        self.db_path = db_path

    def setup(self):
       
        paths.ensure_dirs()
        artifact_manager.initialize_artifacts()
        database.initialize_database(self.db_path)

        split_loader.require_splits()  # load-only: a run can never regenerate splits

        tracking_uri, experiment_name = mlflow_logger.resolve_mlflow_settings(self.configs)
        mlflow_logger.setup_mlflow(tracking_uri, experiment_name)
        return self

    def load_data(self):
        data_cfg = self.experiment_cfg.get("data", {})
        batch_size = (
            self.experiment_cfg.get("training", {}).get("batch_size")
            or self.dataset_cfg.get("dataloader", {}).get("batch_size")
        )
        loaders, meta = split_loader.get_dataloaders(
            self.dataset_cfg,
            batch_size=batch_size,
            max_train_samples=data_cfg.get("max_train_samples"),
            max_val_samples=data_cfg.get("max_val_samples"),
            max_test_samples=data_cfg.get("max_test_samples"),
            nrows=data_cfg.get("nrows"),
        )
        return loaders, meta

    def run(self, verbose=True):
        algorithm_name = self.experiment_cfg.get("algorithm", "random_search")
        if algorithm_name not in ALGORITHMS:
            raise ValueError(f"Unknown algorithm '{algorithm_name}'.")

        self.setup()
        loaders, meta = self.load_data()
        search_space = SearchSpace(self.configs["search_space"])

        seeds = self.experiment_cfg["seeds"]
        budget = int(self.experiment_cfg.get("budget", {}).get("evaluations", 10))

        experiment_id = str(uuid.uuid4())
        database.insert_experiment(
            experiment_id=experiment_id,
            experiment_name=self.experiment_cfg.get("experiment_name"),
            algorithm=algorithm_name,
            num_seeds=len(seeds),
            budget=budget,
            config=self.configs,
            system_info=get_system_info(),
            db_path=self.db_path,
        )

        runner = Runner(
            experiment_id, self.configs, search_space, loaders, meta,
            device=self.device, db_path=self.db_path,
        )

        summaries = []
        for i, seed in enumerate(seeds, start=1):
            if verbose:
                print(f"\n=== Run {i}/{len(seeds)} | {algorithm_name} | seed {seed} | device {self.device} ===")
            summaries.append(runner.run_seed(seed, verbose=verbose))

        exports = csv_exporter.export_all(db_path=self.db_path)
        if verbose:
            print(f"\nExported: {exports['runs_csv']}")
            print(f"Exported: {exports['evaluations_csv']}")

        return {
            "experiment_id": experiment_id,
            "summaries": summaries,
            "exports": exports,
        }
