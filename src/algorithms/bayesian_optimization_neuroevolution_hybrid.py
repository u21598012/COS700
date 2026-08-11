"""Bayesian Optimization + Neuroevolution hybrid search.

``search`` is composed of *phases*. Each phase has the uniform signature
``phase(evaluate_fn, on_evaluation) -> None`` and shares one set of tracking
state on the instance (``_outcome``, ``_eval_count``, ``_best_score``,
``_start``), so evaluation numbering, best-tracking and history run continuously
across the whole search instead of restarting per phase. The pipeline lives in
``self.phases`` and can be reordered, dropped or extended without touching
``search``.

The Bayesian phase (Optuna/TPE) additionally records every successful trial into
``self.bay_opt_outcomes``, in the exact shape
``Neuroevolution.population_metadata`` uses, so a following neuroevolution phase
can seed its population from it.
"""

import time

import optuna

from src.algorithms.base_algorithm import BaseAlgorithm, SearchOutcome
from src.models.architecture import ArchitectureConfig, LayerConfig

optuna.logging.set_verbosity(optuna.logging.WARNING)

import pandas as pd


class BayesianOptimizationNeuroevolutionHybrid(BaseAlgorithm):
    name = "bayesian_optimization_neuroevolution_hybrid"

    def __init__(self, search_space, budget, selection_metric="macro_f1",
                 bo_fraction=0.4):
        super().__init__(search_space, budget, selection_metric)
        # Share of the total evaluation budget spent on the Bayesian phase.
        self.bo_fraction = bo_fraction
        # Per-instance: a class-level list would be shared by every instance
        # (the same trap called out in Neuroevolution.__init__).
        self.bay_opt_outcomes = []
        # Pluggable pipeline. Pass phases=[...] -- bound methods, or any callable
        # taking (evaluate_fn, on_evaluation) -- to recompose the search.
        self._reset_tracking()

    def search(self, evaluate_fn, on_evaluation=None):
        self._reset_tracking()

        bayesian_phase = self.bayesian_optimization_phase(evaluate_fn, on_evaluation, self.budget * self.bo_fraction)
        population = []
        self._outcome.total_search_time = time.perf_counter() - self._start
        dataset = self.create_dataset(self.bay_opt_outcomes)
        return self._outcome
    # ------------------------------------------------------------------
    # Shared state / helpers every phase uses
    # ------------------------------------------------------------------

    def _reset_tracking(self):
        """Reset the bookkeeping for a fresh search. Mirrors Neuroevolution."""
        self._outcome = SearchOutcome()
        self._eval_count = 0
        self._best_score = float("-inf")
        self._start = time.perf_counter()
        self.bay_opt_outcomes = []

    @property
    def remaining_budget(self):
        """Evaluations still available to the current phase."""
        return max(0, self.budget - self._eval_count)

    def _record_evaluation(self, architecture, result, on_evaluation=None):
        """Fold one evaluation into the shared outcome and return its score.

        Mirrors Neuroevolution.evaluate_architecture / BaseAlgorithm._run_loop so
        every phase produces identical numbering, best-tracking, history and
        Runner logging.
        """
        self._eval_count += 1
        elapsed = time.perf_counter() - self._start
        score = self._score(result)

        if score > self._best_score:
            self._best_score = score
            self._outcome.best_architecture = architecture
            self._outcome.best_result = result
            self._outcome.best_evaluation_number = self._eval_count
            self._outcome.time_to_best = elapsed

        self._outcome.history.append({
            "evaluation_number": self._eval_count,
            "score": score,
            "elapsed": elapsed,
        })

        if on_evaluation is not None:
            on_evaluation(self._eval_count, architecture, result)

        return score

    def _load_bounds(self):
        """Flatten the SearchSpace config into the scalars a phase samples from."""
        space = self.search_space.space
        global_cfg = space["global"]
        layer_cfg = space["layer"]

        return {
            "lr_min": float(global_cfg["learning_rate"]["min"]),
            "lr_max": float(global_cfg["learning_rate"]["max"]),
            "layers_min": int(global_cfg["num_layers"]["min"]),
            "layers_max": int(global_cfg["num_layers"]["max"]),
            "hs_min": int(layer_cfg["hidden_size"]["min"]),
            "hs_max": int(layer_cfg["hidden_size"]["max"]),
            "dr_min": float(layer_cfg["dropout"]["min"]),
            "dr_max": float(layer_cfg["dropout"]["max"]),
            "layer_types": layer_cfg["type"]["choices"],
            "activations": layer_cfg.get("activation", {}).get("choices"),
            "kernel_sizes": layer_cfg.get("kernel_size", {}).get("choices"),
            "bidirectionals": layer_cfg.get("bidirectional", {}).get("choices"),
        }

    # ------------------------------------------------------------------
    # Bayesian optimization phase
    # ------------------------------------------------------------------

    def _suggest_architecture(self, trial, b):
        """Build one ArchitectureConfig from an Optuna trial's suggestions."""
        lr = trial.suggest_float("learning_rate", b["lr_min"], b["lr_max"], log=True)
        num_layers = trial.suggest_int("num_layers", b["layers_min"], b["layers_max"])

        layers = []
        for i in range(num_layers):
            layer_type = trial.suggest_categorical(f"layer_{i}_type", b["layer_types"])
            hidden_size = trial.suggest_int(f"layer_{i}_hidden_size", b["hs_min"], b["hs_max"])
            dropout = trial.suggest_float(f"layer_{i}_dropout", b["dr_min"], b["dr_max"])

            activation = None
            if b["activations"]:
                activation = trial.suggest_categorical(f"layer_{i}_activation", b["activations"])

            kernel_size = None
            if b["kernel_sizes"] and layer_type == "CNN":
                kernel_size = trial.suggest_categorical(
                    f"layer_{i}_kernel_size", [int(k) for k in b["kernel_sizes"]]
                )

            bidirectional = None
            if b["bidirectionals"] and layer_type in ("LSTM", "GRU"):
                bidirectional = trial.suggest_categorical(
                    f"layer_{i}_bidirectional", b["bidirectionals"]
                )

            layers.append(LayerConfig(
                layer_type=layer_type,
                hidden_size=hidden_size,
                dropout=dropout,
                activation=activation,
                kernel_size=kernel_size,
                bidirectional=bidirectional,
            ))

        return ArchitectureConfig(
            learning_rate=lr,
            num_layers=num_layers,
            layers=layers,
        )

    def bayesian_optimization_phase(self, evaluate_fn, on_evaluation=None, n_trials=None):
        """Run the Optuna/TPE phase, seeding self.bay_opt_outcomes as it goes."""
        if n_trials is None:
            n_trials = max(1, int(self.budget * self.bo_fraction))
        # Never overspend the shared budget, whatever earlier phases used.
        n_trials = min(int(n_trials), self.remaining_budget)
        if n_trials <= 0:
            return

        bounds = self._load_bounds()

        def objective(trial):
            architecture = self._suggest_architecture(trial, bounds)
            result = evaluate_fn(architecture)
            score = self._record_evaluation(architecture, result, on_evaluation)

            # Seed material for the neuroevolution phase. Same shape as
            # Neuroevolution.population_metadata entries, so these can be handed
            # straight to the NE population (fitness is the full
            # EvaluationResult, whose .macro_f1 NE selection reads). Failed
            # trials are still logged/counted above, just not seeded.
            if result is not None and not result.failed:
                self.bay_opt_outcomes.append({
                    "id": id(architecture),
                    "architecture": architecture,
                    "fitness": result,
                })

            return -score

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    def create_dataset(outcomes):
        df = pd.DataFrame()

    def parse_architecture(architecture_record):
        data = {}
        fitness = architecture_record["fitness"]


if __name__ == "__main__":
    import sys

    from src.evaluation.evaluator import evaluate_architecture
    from src.experiment.experiment_manager import ExperimentManager
    from src.models.search_space import SearchSpace

    # Mirror everything printed during the run to a text file as well as the
    # console (same pattern as neuroevolution.py's __main__).
    class _Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for s in self.streams:
                s.write(data)

        def flush(self):
            for s in self.streams:
                s.flush()

    output_path = "bayesian_optimization_neuroevolution_hybrid_output.txt"
    log_file = open(output_path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_file)

    try:
        experiment_manager = ExperimentManager()

        # Load the dataset ONCE, up front -- not on every evaluation.
        loaders, meta = experiment_manager.load_data()

        # For a fast smoke run, uncomment to cap work per evaluation:
        # experiment_manager.experiment_cfg["training"]["max_train_batches"] = 2
        # experiment_manager.experiment_cfg["training"]["max_eval_batches"] = 2
        # experiment_manager.experiment_cfg["training"]["epochs"] = 1

        def evaluate_fn(architecture):
            print(f"    training {architecture.num_layers}-layer model "
                  f"(lr={architecture.learning_rate:.2e})...", flush=True)
            return evaluate_architecture(
                architecture, loaders["train"], loaders["val"],
                experiment_manager.experiment_cfg, meta["input_dim"],
                meta["num_classes"], device=experiment_manager.device,
            )

        def on_evaluation(eval_number, architecture, result):
            status = (f"FAILED: {result.error}" if result.failed
                      else f"macro_f1={result.macro_f1:.4f} acc={result.accuracy:.4f}")
            print(f"[eval {eval_number}] -> {status}  ({result.training_time:.1f}s train)",
                  flush=True)

        search_space = SearchSpace.from_yaml()
        hybrid = BayesianOptimizationNeuroevolutionHybrid(search_space, budget=10)

        print("=" * 80)
        print(f"Hybrid search: budget={hybrid.budget}, bo_fraction={hybrid.bo_fraction} "
              f"-> {int(hybrid.budget * hybrid.bo_fraction)} Bayesian trials")
        print("=" * 80, "\n")

        outcome = hybrid.search(evaluate_fn, on_evaluation)

        # The seed material the neuroevolution phase will consume.
        print("\n" + "=" * 80)
        print(f"bay_opt_outcomes ({len(hybrid.bay_opt_outcomes)} seeds, "
              f"failed trials excluded)")
        print("=" * 80)
        for seed in sorted(hybrid.bay_opt_outcomes,
                           key=lambda s: s["fitness"].macro_f1, reverse=True):
            print(f"  id={seed['id']}  macro_f1={seed['fitness'].macro_f1:.4f}")
            print(f"    {seed['architecture']}\n")

        print("=" * 80)
        print(f"Best architecture (eval #{outcome.best_evaluation_number}): "
              f"{outcome.best_architecture}")
        print(f"Best fitness: {outcome.best_result}")
        print(f"Time to best: {outcome.time_to_best:.1f}s")
        print(f"Total evaluations: {len(outcome.history)}, "
              f"total search time: {outcome.total_search_time:.1f}s")
    finally:
        sys.stdout = sys.__stdout__
        log_file.close()
        print(f"Output written to {output_path}")

