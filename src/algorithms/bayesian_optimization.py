"""Bayesian Optimization search via Optuna (TPE sampler).

Optuna owns the suggest-evaluate loop. Each trial proposes a full
ArchitectureConfig by sampling from the same bounds the SearchSpace exposes,
then calls the shared evaluate_fn so training/logging are identical to every
other algorithm.
"""

import time

import optuna

from src.algorithms.base_algorithm import BaseAlgorithm, SearchOutcome
from src.models.architecture import ArchitectureConfig, LayerConfig

optuna.logging.set_verbosity(optuna.logging.WARNING)


class BayesianOptimization(BaseAlgorithm):
    name = "bayesian_optimization"

    def search(self, evaluate_fn, on_evaluation=None):
        space = self.search_space.space
        global_cfg = space["global"]
        layer_cfg = space["layer"]

        lr_min = float(global_cfg["learning_rate"]["min"])
        lr_max = float(global_cfg["learning_rate"]["max"])
        layers_min = int(global_cfg["num_layers"]["min"])
        layers_max = int(global_cfg["num_layers"]["max"])

        hs_min = int(layer_cfg["hidden_size"]["min"])
        hs_max = int(layer_cfg["hidden_size"]["max"])
        dr_min = float(layer_cfg["dropout"]["min"])
        dr_max = float(layer_cfg["dropout"]["max"])
        layer_types = layer_cfg["type"]["choices"]
        activations = layer_cfg.get("activation", {}).get("choices")
        kernel_sizes = layer_cfg.get("kernel_size", {}).get("choices")
        bidirectionals = layer_cfg.get("bidirectional", {}).get("choices")

        outcome = SearchOutcome()
        best_score = float("-inf")
        eval_counter = [0]
        start = time.perf_counter()

        def objective(trial):
            nonlocal best_score

            lr = trial.suggest_float("learning_rate", lr_min, lr_max, log=True)
            num_layers = trial.suggest_int("num_layers", layers_min, layers_max)

            layers = []
            for i in range(num_layers):
                layer_type = trial.suggest_categorical(f"layer_{i}_type", layer_types)
                hidden_size = trial.suggest_int(f"layer_{i}_hidden_size", hs_min, hs_max)
                dropout = trial.suggest_float(f"layer_{i}_dropout", dr_min, dr_max)

                activation = None
                if activations:
                    activation = trial.suggest_categorical(f"layer_{i}_activation", activations)

                kernel_size = None
                if kernel_sizes and layer_type == "CNN":
                    kernel_size = trial.suggest_categorical(f"layer_{i}_kernel_size", [int(k) for k in kernel_sizes])

                bidirectional = None
                if bidirectionals and layer_type in ("LSTM", "GRU"):
                    bidirectional = trial.suggest_categorical(f"layer_{i}_bidirectional", bidirectionals)

                layers.append(LayerConfig(
                    layer_type=layer_type,
                    hidden_size=hidden_size,
                    dropout=dropout,
                    activation=activation,
                    kernel_size=kernel_size,
                    bidirectional=bidirectional,
                ))

            architecture = ArchitectureConfig(
                learning_rate=lr,
                num_layers=num_layers,
                layers=layers,
            )

            result = evaluate_fn(architecture)
            score = self._score(result)

            eval_counter[0] += 1
            elapsed = time.perf_counter() - start

            if score > best_score:
                best_score = score
                outcome.best_architecture = architecture
                outcome.best_result = result
                outcome.best_evaluation_number = eval_counter[0]
                outcome.time_to_best = elapsed

            outcome.history.append({
                "evaluation_number": eval_counter[0],
                "score": score,
                "elapsed": elapsed,
            })

            if on_evaluation is not None:
                on_evaluation(eval_counter[0], architecture, result)

            return -score

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(),
        )
        study.optimize(objective, n_trials=self.budget, show_progress_bar=False)

        outcome.total_search_time = time.perf_counter() - start
        return outcome
