

import uuid

from src.algorithms import get_algorithm
from src.evaluation.evaluator import evaluate_architecture
from src.evaluation.metrics import get_confusion_matrix
from src.models.architecture_builder import build_model
from src.tracking import artifact_manager, database, mlflow_logger
from src.utils.seed import set_seed


class Runner:
    def __init__(self, experiment_id, configs, search_space, loaders, meta,
                 device="cpu", db_path=None):
        self.experiment_id = experiment_id
        self.configs = configs
        self.experiment_cfg = configs["experiment"]
        self.search_space = search_space
        self.loaders = loaders
        self.meta = meta
        self.device = device
        self.db_path = db_path

        self.algorithm_name = self.experiment_cfg.get("algorithm", "random_search")
        self.budget = int(self.experiment_cfg.get("budget", {}).get("evaluations", 10))
        self.selection_metric = self.experiment_cfg.get("selection_metric", "macro_f1")

       
        self.algorithm_params = dict(
            (self.experiment_cfg.get("algorithm_params") or {}).get(
                self.algorithm_name, {}
            )
        )

    def run_seed(self, seed, verbose=True):
        set_seed(seed)
        run_id = str(uuid.uuid4())

        algorithm = get_algorithm(
            self.algorithm_name, self.search_space, self.budget,
            selection_metric=self.selection_metric, **self.algorithm_params,
        )
       
        algorithm.set_run_context(run_id, seed)

        with mlflow_logger.start_run(run_name=f"{self.algorithm_name}_seed{seed}") as active:
            mlflow_run_id = active.info.run_id
            database.create_run(
                run_id, self.experiment_id, self.algorithm_name, seed,
                mlflow_run_id=mlflow_run_id, db_path=self.db_path,
            )
            mlflow_logger.log_params({
                "algorithm": self.algorithm_name,
                "seed": seed,
                "budget": self.budget,
                "epochs": self.experiment_cfg.get("training", {}).get("epochs"),
              
                **{f"algo_{k}": v for k, v in self.algorithm_params.items()},
            })

            def evaluate_fn(architecture):
                return evaluate_architecture(
                    architecture, self.loaders["train"], self.loaders["val"],
                    self.experiment_cfg, self.meta["input_dim"],
                    self.meta["num_classes"], device=self.device,
                )

            def on_evaluation(eval_number, architecture, result):
                self._log_evaluation(run_id, seed, eval_number, architecture, result)
                if verbose:
                    flag = " (FAILED)" if result.failed else ""
                    print(
                        f"  [seed {seed}] eval {eval_number}/{self.budget} "
                        f"f1={result.macro_f1:.4f} acc={result.accuracy:.4f}{flag}"
                    )

            outcome = algorithm.search(evaluate_fn, on_evaluation)
            summary = self._finalize(run_id, seed, outcome)

        return summary

    def _log_evaluation(self, run_id, seed, eval_number, architecture, result):
        arch_json = architecture.to_json()
        database.insert_evaluation(
            str(uuid.uuid4()), run_id, self.algorithm_name, seed, eval_number,
            arch_json, result, db_path=self.db_path,
        )
        artifact_manager.save_architecture(
            architecture, f"{run_id}_candidate_{eval_number:04d}"
        )
        mlflow_logger.log_metrics({
            "accuracy": result.accuracy,
            "precision": result.precision,
            "recall": result.recall,
            "macro_f1": result.macro_f1,
            "training_time": result.training_time,
            "peak_memory_mb": result.peak_memory_mb,
        }, step=eval_number)

    def _finalize(self, run_id, seed, outcome):
        best = outcome.best_result
        best_arch = outcome.best_architecture
        best_arch_json = best_arch.to_json() if best_arch is not None else None

        artifact_manager.save_search_progress(outcome.history, f"{run_id}_search_progress")

        model_path = None
        if best_arch is not None:
           
            artifact_manager.save_architecture(best_arch, f"{run_id}_best")
            model = build_model(best_arch, self.meta["input_dim"], self.meta["num_classes"])
            model_path = str(artifact_manager.save_model(model, f"{run_id}_best"))

            if best is not None and best.predictions is not None:
                artifact_manager.save_predictions(
                    best.targets, best.predictions, f"{run_id}_best_predictions"
                )
                cm = get_confusion_matrix(
                    best.targets, best.predictions, self.meta["num_classes"]
                )
                artifact_manager.save_confusion_matrix(
                    cm, f"{run_id}_best_confusion_matrix",
                    class_names=list(range(self.meta["num_classes"])),
                )

        summary = {
            "run_id": run_id,
            "seed": seed,
            "total_search_time": outcome.total_search_time,
            "time_to_best": outcome.time_to_best,
            "best_evaluation_number": outcome.best_evaluation_number,
            "best_accuracy": best.accuracy if best else None,
            "best_precision": best.precision if best else None,
            "best_recall": best.recall if best else None,
            "best_macro_f1": best.macro_f1 if best else None,
            "peak_memory_mb": best.peak_memory_mb if best else None,
            "best_architecture": best_arch_json,
        }

        database.finalize_run(run_id, summary, db_path=self.db_path)
        if best_arch is not None:
            database.insert_best_model(
                run_id, outcome.best_evaluation_number, best_arch_json,
                summary["best_accuracy"], summary["best_macro_f1"], model_path,
                db_path=self.db_path,
            )
        mlflow_logger.log_metrics({
            "best_accuracy": summary["best_accuracy"] or 0.0,
            "best_macro_f1": summary["best_macro_f1"] or 0.0,
            "total_search_time": summary["total_search_time"],
            "time_to_best": summary["time_to_best"],
        })
        return summary
