import copy
import itertools
import math
import random
import time
from pathlib import Path

import optuna

from src.algorithms.base_algorithm import BaseAlgorithm, SearchOutcome
from src.evaluation.evaluator import EvaluationResult
from src.models.architecture import ArchitectureConfig, LayerConfig
from src.utils import paths

optuna.logging.set_verbosity(optuna.logging.WARNING)

import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, LeaveOneOut, train_test_split
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

class BayesianOptimizationNeuroevolutionHybrid(BaseAlgorithm):
    name = "bayesian_optimization_neuroevolution_hybrid"
    SURROGATE_TRAINERS = {
        "xgboost": "train_xgboost_model",
        "ridge": "train_ridge_regression_model",
        "svm": "train_svm",
        "elastic_net": "train_elastic_net_log_regression",
    }
    MIN_SURROGATE_ROWS = 2

    MAX_BARREN_GENERATIONS = 3

    def __init__(self, search_space, budget, selection_metric="macro_f1",
                 bo_fraction=0.4, num_selections=10, mutation_rate=1.0,
                 crossover_rate=1.0, num_final_selections=2,
                 surrogate="xgboost", verbose=False, dataset_path=None):
        super().__init__(search_space, budget, selection_metric)
        self.dataset_path = Path(dataset_path) if dataset_path is not None else None

        if surrogate not in self.SURROGATE_TRAINERS:
            raise ValueError(
                f"Unknown surrogate '{surrogate}'. "
                f"Available: {sorted(self.SURROGATE_TRAINERS)}"
            )
        self.surrogate = surrogate
        self.bo_fraction = bo_fraction

        self.bo_evaluations = max(1, int(round(budget * bo_fraction)))
        self.population_size = self.bo_evaluations
        self.num_final_selections = max(1, int(num_final_selections))
        remaining = max(0, self.budget - self.population_size)
        self.generation_size = math.ceil(remaining / self.num_final_selections)

        self.num_selections = max(2, int(num_selections))
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        self.surrogate_model = None

        self.verbose = verbose
        self._reset_tracking()

    def set_run_context(self, run_id, seed):
        self.dataset_path = paths.surrogate_dataset_path(
            f"{self.name}_seed{seed}_{str(run_id)[:8]}"
        )
        return self

    def search(self, evaluate_fn, on_evaluation=None):
        self._reset_tracking()

        self._log("=" * 78)
        self._log(f"PHASE 1 - Bayesian optimization: {self.bo_evaluations} trials")
        self._log("=" * 78)
        self.bayesian_optimization_phase(evaluate_fn, on_evaluation,
                                         self.bo_evaluations)

        dataset = self.create_dataset(self.evaluation_records)
        
        saved_to = self.save_dataset(dataset)
        self._log(f"\nSurrogate dataset ({len(dataset)} rows) saved to {saved_to}\n")

        if len(dataset) < self.MIN_SURROGATE_ROWS:
            return self._abandon_search(
                f"the Bayesian phase produced only {len(dataset)} usable "
                f"result(s) from {self._eval_count} evaluation(s); the "
                f"neuroevolution phase needs at least "
                f"{self.MIN_SURROGATE_ROWS} to seed a population and fit a "
                f"surrogate"
            )

        self.install_surrogate(self.train_surrogate(dataset))

        self._log("\n" + "=" * 78)
        self._log(f"PHASE 2 - Neuroevolution: {self.generation_size} generations, "
                  f"{self.remaining_budget} evaluations remaining")
        self._log("=" * 78)
        self.neuroevolution_phase(evaluate_fn, on_evaluation)

        saved_to = self.save_dataset(self.create_dataset(self.evaluation_records))
        self._log(f"\nFinal surrogate dataset ({len(self.evaluation_records)} rows) "
                  f"saved to {saved_to}")

        self._outcome.total_search_time = time.perf_counter() - self._start
        return self._outcome

    def _reset_tracking(self):
        self._outcome = SearchOutcome()
        self._eval_count = 0
        self._best_score = float("-inf")
        self._start = time.perf_counter()
        self.bay_opt_outcomes = []
       
        self.evaluation_records = []
       
        self.population = []
        self.population_metadata = {}
        self.best_architecture = None
    
        self.surrogates = {}
        self.surrogate_reports = {}
        self.surrogate_model = None

    def _log(self, *args, **kwargs):
        print(*args, **kwargs)

    def _abandon_search(self, reason):
        self._log(f"\nSEARCH ABANDONED: {reason}.")
        self._log("Reporting zero for every performance metric.")

        self._outcome.best_architecture = None
        self._outcome.best_result = EvaluationResult(failed=True, error=reason)
        self._outcome.best_evaluation_number = None
        self._outcome.time_to_best = 0.0
        self._outcome.total_search_time = time.perf_counter() - self._start
        return self._outcome

    @property
    def remaining_budget(self):
        return max(0, self.budget - self._eval_count)

    def _record_evaluation(self, architecture, result, on_evaluation=None):
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

     
        if result is not None and not result.failed:
            self.evaluation_records.append({
                "id": id(architecture),
                "architecture": architecture,
                "fitness": result,
            })

        if on_evaluation is not None:
            on_evaluation(self._eval_count, architecture, result)

        return score

    def _load_bounds(self):
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
            "poolings": layer_cfg.get("pooling", {}).get("choices"),
        }

    def _suggest_architecture(self, trial, b):
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

      
            pooling = None
            if b["poolings"]:
                pooling = trial.suggest_categorical(f"layer_{i}_pooling", b["poolings"])

            layers.append(LayerConfig(
                layer_type=layer_type,
                hidden_size=hidden_size,
                dropout=dropout,
                activation=activation,
                kernel_size=kernel_size,
                bidirectional=bidirectional,
                pooling=pooling,
            ))

        return ArchitectureConfig(
            learning_rate=lr,
            num_layers=num_layers,
            layers=layers,
        )

    def bayesian_optimization_phase(self, evaluate_fn, on_evaluation=None, n_trials=None):
        if n_trials is None:
            n_trials = max(1, int(self.budget * self.bo_fraction))
        n_trials = min(int(n_trials), self.remaining_budget)
        if n_trials <= 0:
            return

        bounds = self._load_bounds()

        def objective(trial):
            architecture = self._suggest_architecture(trial, bounds)
            result = evaluate_fn(architecture)
            score = self._record_evaluation(architecture, result, on_evaluation)

         
            if result is not None and not result.failed:
                self.bay_opt_outcomes.append({
                    "id": id(architecture),
                    "architecture": architecture,
                    "fitness": result,
                })

            return -score

        study = optuna.create_study(
            direction="minimize",
          
            sampler=optuna.samplers.TPESampler(seed=random.randrange(2 ** 31)),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    FEATURE_COLUMNS = [
        "num_layers",
        "log_learning_rate",
        "gru_count",
        "lstm_count",
        "cnn_count",
        "avg_hidden_size",
        "avg_dropout",
        "gru_avg_hidden_size",
        "lstm_avg_hidden_size",
        "cnn_avg_hidden_size",
        "gru_avg_dropout",
        "lstm_avg_dropout",
        "cnn_avg_dropout",
        "gru_bidirectional_count",
        "lstm_bidirectional_count",
        "gru_relu_count",
        "gru_tanh_count",
        "gru_elu_count",
        "lstm_relu_count",
        "lstm_tanh_count",
        "lstm_elu_count",
     
        "cnn_relu_count",
        "cnn_tanh_count",
        "cnn_elu_count",
      
        "cnn_kernel_3_count",
        "cnn_kernel_5_count",
        "cnn_kernel_7_count",
      
        "max_pool_count",
        "avg_pool_count",
    ]
    TARGET_COLUMN = "macro_f1"

    def create_dataset(self, outcomes=None):

        if outcomes is None:
            outcomes = self.bay_opt_outcomes

        rows = [self.parse_architecture(record) for record in outcomes]
        return pd.DataFrame(rows, columns=self.FEATURE_COLUMNS + [self.TARGET_COLUMN])

    def save_dataset(self, dataset, path=None):
       
        if path is None:
            path = self.dataset_path or paths.SURROGATE_DATASET_CSV
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(path, index=False)
        return path

    def load_dataset(self, path=None):
       
        if path is None:
            path = self.dataset_path or paths.SURROGATE_DATASET_CSV
        path = Path(path)
        dataset = pd.read_csv(path)

        expected = self.FEATURE_COLUMNS + [self.TARGET_COLUMN]
        missing = [c for c in expected if c not in dataset.columns]
        extra = [c for c in dataset.columns if c not in expected]
        if missing or extra:
            raise ValueError(
                f"{path} does not match the current feature list "
                f"(missing={missing}, unexpected={extra}). It was written by an "
                f"older version of parse_architecture -- delete it and re-run the "
                f"search to regenerate it."
            )
        return dataset[expected]

    def parse_architecture(self, architecture_record):
        row = self._feature_row(architecture_record["architecture"])
        row[self.TARGET_COLUMN] = float(architecture_record["fitness"].macro_f1)
        return row

    def _feature_row(self, architecture):
       
        layers = architecture.layers or []

        gru = [l for l in layers if l.layer_type == "GRU"]
        lstm = [l for l in layers if l.layer_type == "LSTM"]
        cnn = [l for l in layers if l.layer_type == "CNN"]

        def mean(values):
           
            return float(sum(values) / len(values)) if values else 0.0

        def activation_count(type_layers, activation):
            return sum(1 for l in type_layers if l.activation == activation)

        learning_rate = max(float(architecture.learning_rate), 1e-12)

        data = {

            "num_layers": len(layers),
            "log_learning_rate": math.log10(learning_rate),

            "gru_count": len(gru),
            "lstm_count": len(lstm),
            "cnn_count": len(cnn),

            "avg_hidden_size": mean([l.hidden_size for l in layers]),
            "avg_dropout": mean([l.dropout for l in layers]),

            "gru_avg_hidden_size": mean([l.hidden_size for l in gru]),
            "lstm_avg_hidden_size": mean([l.hidden_size for l in lstm]),
            "cnn_avg_hidden_size": mean([l.hidden_size for l in cnn]),

            "gru_avg_dropout": mean([l.dropout for l in gru]),
            "lstm_avg_dropout": mean([l.dropout for l in lstm]),
            "cnn_avg_dropout": mean([l.dropout for l in cnn]),

            "gru_bidirectional_count": sum(1 for l in gru if l.bidirectional),
            "lstm_bidirectional_count": sum(1 for l in lstm if l.bidirectional),

            "gru_relu_count": activation_count(gru, "relu"),
            "gru_tanh_count": activation_count(gru, "tanh"),
            "gru_elu_count": activation_count(gru, "elu"),
            "lstm_relu_count": activation_count(lstm, "relu"),
            "lstm_tanh_count": activation_count(lstm, "tanh"),
            "lstm_elu_count": activation_count(lstm, "elu"),
            "cnn_relu_count": activation_count(cnn, "relu"),
            "cnn_tanh_count": activation_count(cnn, "tanh"),
            "cnn_elu_count": activation_count(cnn, "elu"),

            "cnn_kernel_3_count": sum(1 for l in cnn if l.kernel_size == 3),
            "cnn_kernel_5_count": sum(1 for l in cnn if l.kernel_size == 5),
            "cnn_kernel_7_count": sum(1 for l in cnn if l.kernel_size == 7),

            "max_pool_count": sum(1 for l in layers if l.pooling == "max"),
            "avg_pool_count": sum(1 for l in layers if l.pooling == "avg"),
        }
        return data

    def _feature_frame(self, architectures):
        rows = [self._feature_row(a) for a in architectures]
        return pd.DataFrame(rows, columns=self.FEATURE_COLUMNS)

    def install_surrogate(self, model):
      
        self.surrogate_model = model
        return model

    def predict_fitness(self, architectures):
       
        if self.surrogate_model is None:
            raise RuntimeError(
                "No surrogate has been fitted yet -- call train_surrogate and "
                "install_surrogate (or run the Bayesian phase) before scoring "
                "with the surrogate."
            )
        if not architectures:
            return []
        return [float(p) for p in self.surrogate_model.predict(self._feature_frame(architectures))]

    def _report_fit(self, label, model, X_train, y_train, X_test, y_test,
                    grid_search=None):
       
        train_pred = model.predict(X_train)
        train_rmse = math.sqrt(mean_squared_error(y_train, train_pred))
        train_mae = mean_absolute_error(y_train, train_pred)
        baseline_rmse = math.sqrt(
            mean_squared_error(y_train, np.full(len(y_train), y_train.mean()))
        )

        report = {
            "label": label,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "train_rmse": train_rmse,
            "train_mae": train_mae,
            "train_r2": None,
            "baseline_rmse": baseline_rmse,
            "cv_rmse": None,
            "test_rmse": None,
            "best_params": None,
            "beats_baseline": None,
        }

        print(f"  [{label}] n_train={len(X_train)} n_test={len(X_test)}")
        line = f"      train  RMSE={train_rmse:.5f}  MAE={train_mae:.5f}"
        if len(y_train) >= 2 and y_train.var() > 0:
            report["train_r2"] = r2_score(y_train, train_pred)
            line += f"  R2={report['train_r2']:+.4f}"
        print(line)
        print(f"      mean-baseline RMSE={baseline_rmse:.5f}"
              f"   (train RMSE below this only proves it can memorise)")

        if grid_search is not None:
            cv_rmse = math.sqrt(-grid_search.best_score_)
            report["cv_rmse"] = cv_rmse
            report["best_params"] = grid_search.best_params_
            report["beats_baseline"] = cv_rmse < baseline_rmse
            verdict = "beats baseline" if report["beats_baseline"] else "NO BETTER THAN THE MEAN"
            print(f"      LOO-CV RMSE={cv_rmse:.5f}  -> {verdict}")
            print(f"      best params: {grid_search.best_params_}")

        if len(X_test) > 0:
            test_pred = model.predict(X_test)
            report["test_rmse"] = math.sqrt(mean_squared_error(y_test, test_pred))
            print(f"      held-out RMSE={report['test_rmse']:.5f} on {len(X_test)} row(s)"
                  f"   (too few rows to read much into)")

        return report

    def _tune_regressor(self, dataset, estimator, param_grid, label,
                        test_size=0.2, random_state=42):
        X = dataset[self.FEATURE_COLUMNS]
        y = dataset[self.TARGET_COLUMN]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        if len(X_train) < 3:
            model = estimator.fit(X_train, y_train)
            self.surrogate_reports[label] = self._report_fit(
                label, model, X_train, y_train, X_test, y_test)
            self.surrogates[label] = model
            return model

        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=LeaveOneOut(),
            scoring="neg_mean_squared_error",
            n_jobs=1,
        )
        grid_search.fit(X_train, y_train)

        model = grid_search.best_estimator_
        self.surrogate_reports[label] = self._report_fit(
            label, model, X_train, y_train, X_test, y_test, grid_search)
        self.surrogates[label] = model
        return model

    def train_surrogate(self, dataset, test_size=0.2, random_state=42):
        trainer = getattr(self, self.SURROGATE_TRAINERS[self.surrogate])
        return trainer(dataset, test_size=test_size, random_state=random_state)

    def train_xgboost_model(self, dataset, test_size=0.2, random_state=42):
        model = XGBRegressor(random_state=random_state, eval_metric="rmse")

        param_grid = {
            "max_depth": [1, 2],
            "n_estimators": [50, 200],
            "learning_rate": [0.05, 0.1, 0.3],
            "colsample_bytree": [0.7],
            "reg_lambda": [0.1, 1.0, 5.0],
            "reg_alpha": [0.0],
        }
        return self._tune_regressor(dataset, model, param_grid, "xgboost",
                                    test_size, random_state)

    def train_ridge_regression_model(self, dataset, test_size=0.2, random_state=42):
        pipeline = Pipeline([
            ("scale", StandardScaler()),
            ("ridge", Ridge(random_state=random_state)),
        ])

        param_grid = {
            "ridge__alpha": np.logspace(-3, 3, 13),
            "ridge__fit_intercept": [True, False],
        }
        return self._tune_regressor(dataset, pipeline, param_grid, "ridge",
                                    test_size, random_state)

    def train_svm(self, dataset, test_size=0.2, random_state=42):
        pipeline = Pipeline([
            ("scale", StandardScaler()),
            ("svr", SVR()),
        ])

        param_grid = {
            "svr__kernel": ["rbf", "linear"],
            "svr__C": np.logspace(-2, 3, 6),
            "svr__gamma": ["scale", "auto"],
            "svr__epsilon": [0.001, 0.01, 0.05],
        }
        return self._tune_regressor(dataset, pipeline, param_grid, "svm",
                                    test_size, random_state)

    def train_elastic_net_log_regression(self, dataset, test_size=0.2, random_state=42):
        pipeline = Pipeline([
            ("scale", StandardScaler()),
            ("elastic_net", ElasticNet(random_state=random_state, max_iter=20000)),
        ])

        param_grid = {
            "elastic_net__alpha": np.logspace(-4, 1, 11),
            "elastic_net__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0],
        }
        return self._tune_regressor(dataset, pipeline, param_grid, "elastic_net",
                                    test_size, random_state)

    def _vprint(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def _fitness_of(self, architecture):
        return self.population_metadata[id(architecture)]["fitness"].macro_f1

    def _register(self, architecture, fitness):
        self.population_metadata[id(architecture)] = {
            "id": id(architecture),
            "architecture": architecture,
            "fitness": fitness,
        }

    def seed_population(self, architecture_set=None):
        if architecture_set is None:
            architecture_set = self.bay_opt_outcomes

        self.population = []
        self.population_metadata = {}
        for record in architecture_set:
            architecture = record["architecture"]
            self.population.append(architecture)
            self._register(architecture, record["fitness"])

        self.population_size = len(self.population)
        return self.population

    def evaluate_architecture(self, architecture, evaluate_fn, on_evaluation=None):
        result = evaluate_fn(architecture)
        self._record_evaluation(architecture, result, on_evaluation)
        return result

    def evaluate_and_register(self, architecture, evaluate_fn, on_evaluation=None):
        existing = self.population_metadata.get(id(architecture))
        if existing is not None and existing["fitness"] is not None:
            return existing["fitness"]
        fitness = self.evaluate_architecture(architecture, evaluate_fn, on_evaluation)
        self._register(architecture, fitness)
        return fitness

    def select_best_architecture(self):
        best_architecture = self.best_architecture
        best_fitness = float("-inf")
        if best_architecture is not None and id(best_architecture) in self.population_metadata:
            best_fitness = self._fitness_of(best_architecture)
        else:
            best_architecture = None

        for architecture in self.population:
            fitness = self._fitness_of(architecture)
            if fitness > best_fitness:
                best_fitness = fitness
                best_architecture = architecture

        self.best_architecture = best_architecture
        return best_architecture

    def roulette_wheel_selection(self, num_selections):
        candidates = [self.population_metadata[id(a)] for a in self.population]
        num_selections = min(num_selections, len(candidates))
        selected = []

        for _ in range(num_selections):
            total_fitness = sum(m["fitness"].macro_f1 for m in candidates)
            if total_fitness <= 0:
                chosen = random.choice(candidates)
            else:
                p = random.uniform(0, total_fitness)
                current = 0.0
                chosen = candidates[-1]
                for metadata in candidates:
                    current += metadata["fitness"].macro_f1
                    if current >= p:
                        chosen = metadata
                        break
            selected.append(chosen["architecture"])
            candidates.remove(chosen)

        return selected

    def tournament_selection(self, num_selections):
        candidates = [self.population_metadata[id(a)] for a in self.population]
        num_selections = min(num_selections, len(candidates))
        selected = []
        for _ in range(num_selections):
            tournament_size = min(4, len(candidates))
            tournament = random.sample(candidates, tournament_size)
            best = max(tournament, key=lambda m: m["fitness"].macro_f1)
            selected.append(best["architecture"])
            candidates.remove(best)
        return selected

    def elitist_selection(self, num_selections, scored_architectures):
        ranked = sorted(scored_architectures, key=lambda pair: pair[1], reverse=True)
        return [architecture for architecture, _ in ranked[:num_selections]]

    def mutate_architecture(self, architecture):
        if random.choice(["global", "layer"]) == "global":
            return self.global_mutation(architecture)
        return self.inner_layer_mutation(architecture)

    def inner_layer_mutation(self, architecture):
        arch_ = copy.deepcopy(architecture)
        if not arch_.layers:
            return arch_

        layer_index = random.randrange(len(arch_.layers))
        layer = arch_.layers[layer_index]
        original = copy.deepcopy(layer)
        ss = self.search_space

        if layer.layer_type == "CNN":
            features = ["hidden_size", "dropout", "activation", "kernel_size",
                        "pooling"]
        else:
            features = ["hidden_size", "dropout", "activation", "bidirectional",
                        "pooling"]

        for _ in range(50):
            feature = random.choice(features)
            if feature == "hidden_size":
                layer.hidden_size = ss.sample_hidden_size()
            elif feature == "dropout":
                layer.dropout = ss.sample_dropout()
            elif feature == "activation":
                layer.activation = ss.sample_activation()
            elif feature == "kernel_size":
                layer.kernel_size = ss.sample_kernel_size()
            elif feature == "bidirectional":
                layer.bidirectional = ss.sample_bidirectional()
            elif feature == "pooling":
                layer.pooling = ss.sample_pooling()
            if layer != original:
                break

        arch_.num_layers = len(arch_.layers)
        return arch_

    def create_layer(self, exclude_type=None):
        layer = self.search_space.sample_layer()
        choices = self.search_space.space["layer"]["type"]["choices"]
        if exclude_type is not None and len(choices) > 1:
            for _ in range(50):
                if layer.layer_type != exclude_type:
                    break
                layer = self.search_space.sample_layer()
        return layer

    def global_mutation(self, architecture):
        arch_ = copy.deepcopy(architecture)
        bounds = self.search_space.space["global"]["num_layers"]
        layers_min, layers_max = int(bounds["min"]), int(bounds["max"])

        if random.random() < 0.5:
            arch_.learning_rate = self.search_space.sample_learning_rate()
        else:
            updates = []
            if len(arch_.layers) > 0:
                updates.append("replace")
            if len(arch_.layers) < layers_max:
                updates.append("create")
            if len(arch_.layers) > layers_min:
                updates.append("delete")
            if not updates:
                return arch_

            update = random.choice(updates)
            if update == "create":
                arch_.layers.append(self.create_layer())
            elif update == "replace":
                layer_index = random.randrange(len(arch_.layers))
                current_type = arch_.layers[layer_index].layer_type
                arch_.layers[layer_index] = self.create_layer(exclude_type=current_type)
            elif update == "delete":
                del arch_.layers[random.randrange(len(arch_.layers))]

        arch_.num_layers = len(arch_.layers)
        return arch_

    def _clamp_layers(self, layers):
        bounds = self.search_space.space["global"]["num_layers"]
        layers_min, layers_max = int(bounds["min"]), int(bounds["max"])
        layers = list(layers[:layers_max])
        while len(layers) < layers_min:
            layers.append(self.create_layer())
        return layers

    def _make_child(self, learning_rate, layers):
        layers = self._clamp_layers(copy.deepcopy(layers))
        return ArchitectureConfig(
            learning_rate=learning_rate,
            num_layers=len(layers),
            layers=layers,
        )

    def single_point_crossover(self, parent1, parent2):
        min_layers = min(len(parent1.layers), len(parent2.layers))
        point = random.randint(1, min_layers - 1) if min_layers > 1 else 1

        child_1 = self._make_child(parent1.learning_rate,
                                   parent1.layers[:point] + parent2.layers[point:])
        child_2 = self._make_child(parent2.learning_rate,
                                   parent2.layers[:point] + parent1.layers[point:])
        return child_1, child_2

    def uniform_crossover(self, parent1, parent2):
        min_layers = min(len(parent1.layers), len(parent2.layers))

        child1_layers, child2_layers = [], []
        for i in range(min_layers):
            if random.random() < 0.5:
                child1_layers.append(parent1.layers[i])
                child2_layers.append(parent2.layers[i])
            else:
                child1_layers.append(parent2.layers[i])
                child2_layers.append(parent1.layers[i])

        return (self._make_child(parent1.learning_rate, child1_layers),
                self._make_child(parent2.learning_rate, child2_layers))

    def crossover(self, parent1, parent2):
        if random.choice(["single_point", "uniform"]) == "single_point":
            self._vprint("  single-point crossover")
            child1, child2 = self.single_point_crossover(parent1, parent2)
        else:
            self._vprint("  uniform crossover")
            child1, child2 = self.uniform_crossover(parent1, parent2)
        return [child1, child2]

    def generate_offspring(self, parents):
        offspring = []
        for parent1, parent2 in itertools.combinations(parents, 2):
            if random.random() >= self.crossover_rate:
                continue
            offspring.extend(self.crossover(parent1, parent2))
        return offspring

    def _deduplicate(self, architectures):
        known = list(self.population)
        unique = []
        for architecture in architectures:
            if architecture in known or architecture in unique:
                continue
            unique.append(architecture)
        return unique

    def steady_state_replacement(self, candidates):
        for architecture in candidates:
            if architecture not in self.population:
                self._vprint(f"  + adding {id(architecture)} "
                             f"(macro_f1={self._fitness_of(architecture):.4f})")
                self.population.append(architecture)

        while len(self.population) > self.population_size:
            weakest = min(self.population, key=self._fitness_of)
            self._vprint(f"  - removing {id(weakest)} "
                         f"(macro_f1={self._fitness_of(weakest):.4f})")
            self.population.remove(weakest)

        alive = {id(a) for a in self.population}
        if self.best_architecture is not None:
            alive.add(id(self.best_architecture))
        for key in [k for k in self.population_metadata if k not in alive]:
            self.population_metadata.pop(key)

    def update_surrogate_model(self):
        dataset = self.create_dataset(self.evaluation_records)
        self.save_dataset(dataset)
        self.install_surrogate(self.train_surrogate(dataset))
        return dataset

    def neuroevolution_phase(self, evaluate_fn, on_evaluation=None):
        if not self.population:
            self.seed_population()
        if not self.population:
            self._log("  no seed population -- skipping the neuroevolution phase")
            return self._outcome

        self.select_best_architecture()
        self._log(f"  seeded population: {len(self.population)} architectures, "
                  f"best macro_f1={self._fitness_of(self.best_architecture):.4f}")

        if self.remaining_budget <= 0:
            self._log("  budget exhausted before the neuroevolution phase began "
                      "-- no generations will run")
            return self._outcome

        generation = 0
        barren = 0
        while self.remaining_budget > 0:
            generation += 1

            self._vprint("=" * 78)
            self._log(f"\n-- generation {generation}/{self.generation_size} "
                      f"(budget left: {self.remaining_budget}) --")

            parents = self.roulette_wheel_selection(self.num_selections)
            self._vprint(f"  selected {len(parents)} parents")

            mutated = []
            for parent in parents:
                if random.random() < self.mutation_rate:
                    mutated.append(self.mutate_architecture(parent))
                else:
                    mutated.append(parent)

            offspring = self.generate_offspring(mutated)
            candidates = self._deduplicate(offspring)
            self._log(f"  {len(parents)} parents -> {len(offspring)} offspring "
                      f"-> {len(candidates)} unique unseen")

            if not candidates:
                barren += 1
                self._log(f"  no new candidates this generation "
                          f"({barren}/{self.MAX_BARREN_GENERATIONS})")
                if barren >= self.MAX_BARREN_GENERATIONS:
                    self._log("  population has converged -- no new architecture "
                              "can be produced, ending the phase with "
                              f"{self.remaining_budget} evaluation(s) unspent")
                    break
                continue
            barren = 0

            predictions = self.predict_fitness(candidates)
            scored = list(zip(candidates, predictions))
            self._log(f"  surrogate predictions: best={max(predictions):.4f} "
                      f"worst={min(predictions):.4f}")

            take = min(self.num_final_selections, self.remaining_budget)
            chosen = self.elitist_selection(take, scored)

            evaluated = []
            for architecture in chosen:
                predicted = dict((id(a), p) for a, p in scored)[id(architecture)]
                fitness = self.evaluate_and_register(architecture, evaluate_fn,
                                                     on_evaluation)
                evaluated.append(architecture)
                self._log(f"  real eval #{self._eval_count}: "
                          f"predicted={predicted:.4f} actual={fitness.macro_f1:.4f} "
                          f"(error={abs(predicted - fitness.macro_f1):.4f})")

            self.steady_state_replacement(evaluated)
            self.select_best_architecture()
            self._log(f"  population best macro_f1="
                      f"{self._fitness_of(self.best_architecture):.4f} "
                      f"(population size {len(self.population)})")

            if self.remaining_budget > 0:
                self.update_surrogate_model()

        return self._outcome

if __name__ == "__main__":
    import sys

    from src.evaluation.evaluator import evaluate_architecture
    from src.experiment.experiment_manager import ExperimentManager
    from src.models.search_space import SearchSpace
    from src.utils.seed import set_seed

    BUDGET = 50
    BO_FRACTION = 0.4
    SELECTION_METRIC = "macro_f1"
    SEED = 6011572

    NUM_SELECTIONS = 10
    MUTATION_RATE = 1.0
    CROSSOVER_RATE = 1.0
    NUM_FINAL_SELECTIONS = 2
    VERBOSE = False

    SURROGATE = "xgboost"

    EPOCHS = 10
    BATCH_SIZE = 128
    MAX_TRAIN_BATCHES = None
    MAX_EVAL_BATCHES = None
    MAX_TRAIN_SAMPLES = None
    MAX_VAL_SAMPLES = None

    DATASET_PATH = paths.SURROGATE_DATASET_CSV

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

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 250)

    try:
        set_seed(SEED)

        search_space = SearchSpace.from_yaml()
        hybrid = BayesianOptimizationNeuroevolutionHybrid(
            search_space, budget=BUDGET, selection_metric=SELECTION_METRIC,
            bo_fraction=BO_FRACTION, num_selections=NUM_SELECTIONS,
            mutation_rate=MUTATION_RATE, crossover_rate=CROSSOVER_RATE,
            num_final_selections=NUM_FINAL_SELECTIONS, surrogate=SURROGATE,
            verbose=VERBOSE,
        )

        experiment_manager = ExperimentManager()
        training_cfg = experiment_manager.experiment_cfg.setdefault("training", {})
        training_cfg["epochs"] = EPOCHS
        training_cfg["batch_size"] = BATCH_SIZE
        training_cfg["max_train_batches"] = MAX_TRAIN_BATCHES
        training_cfg["max_eval_batches"] = MAX_EVAL_BATCHES
        data_cfg = experiment_manager.experiment_cfg.setdefault("data", {})
        data_cfg["max_train_samples"] = MAX_TRAIN_SAMPLES
        data_cfg["max_val_samples"] = MAX_VAL_SAMPLES

        loaders, meta = experiment_manager.load_data()

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
            print(f"[eval {eval_number}/{BUDGET}] -> {status}  "
                  f"({result.training_time:.1f}s train)", flush=True)

        print("=" * 80)
        print(f"Hybrid search on {experiment_manager.device}")
        print(f"  budget={BUDGET}  bo_fraction={BO_FRACTION}  seed={SEED}")
        print(f"  phase 1: {hybrid.bo_evaluations} Bayesian trials "
              f"-> population of {hybrid.population_size}")
        print(f"  phase 2: {hybrid.generation_size} generations x "
              f"{NUM_FINAL_SELECTIONS} real evals "
              f"= {hybrid.generation_size * NUM_FINAL_SELECTIONS} evals")
        print(f"  GA: {NUM_SELECTIONS} parents -> "
              f"{NUM_SELECTIONS * (NUM_SELECTIONS - 1) // 2} unique pairs -> "
              f"up to {NUM_SELECTIONS * (NUM_SELECTIONS - 1)} offspring/generation "
              f"(surrogate-scored)")
        print(f"  surrogate: {SURROGATE} (refitted every generation; "
              f"the only model this run fits)")
        print(f"  training: epochs={EPOCHS} batch_size={BATCH_SIZE} "
              f"max_train_batches={MAX_TRAIN_BATCHES} "
              f"max_eval_batches={MAX_EVAL_BATCHES}")
        print(f"  data: max_train_samples={MAX_TRAIN_SAMPLES} "
              f"max_val_samples={MAX_VAL_SAMPLES} "
              f"(train batches: {len(loaders['train'])})")
        print(f"  dataset cached to {DATASET_PATH}")
        print("=" * 80, "\n")

        outcome = hybrid.search(evaluate_fn, on_evaluation)

        abandoned = outcome.best_architecture is None

        dataset = hybrid.create_dataset(hybrid.evaluation_records)

        if not abandoned:
            print("\n" + "=" * 80)
            print(f"Final population ({len(hybrid.population)} architectures)")
            print("=" * 80)
            for architecture in sorted(hybrid.population,
                                       key=hybrid._fitness_of, reverse=True):
                print(f"  macro_f1={hybrid._fitness_of(architecture):.4f}  "
                      f"id={id(architecture)}")
                print(f"    {architecture}\n")

        print("\n" + "=" * 80)
        print(f"Surrogate dataset: {dataset.shape[0]} rows x {dataset.shape[1]} cols "
              f"({len(hybrid.FEATURE_COLUMNS)} features + {hybrid.TARGET_COLUMN} target)")
        print(f"  one row per usable evaluation, both phases "
              f"({len(outcome.history)} evaluations made, "
              f"{len(hybrid.population)} still in the population)")
        print("=" * 80)
        print(dataset.to_string(index=False))

        target = dataset[hybrid.TARGET_COLUMN]
        if len(target) == 0:
            print("\ntarget spread: no rows -- every evaluation failed.")
        else:
            print(f"\ntarget spread: min={target.min():.4f} max={target.max():.4f} "
                  f"mean={target.mean():.4f} std={target.std():.4f} "
                  f"distinct={target.nunique()}/{len(target)}")
            if target.nunique() < 3:
                print("WARNING: nearly every architecture scored the same. Undertrained "
                      "networks all collapse to the majority class, and no surrogate can "
                      "learn from a constant target -- expect the verdicts below to say "
                      "NO BETTER THAN THE MEAN regardless of which model is used.")

        print("\n" + "=" * 80)
        print(f"Surrogate report - {SURROGATE} (leave-one-out CV RMSE, lower is better)")
        print("=" * 80)
        reports = list(hybrid.surrogate_reports.values())
        if not reports:
            print("  no surrogate was fitted")
        else:
            baseline = reports[0]["baseline_rmse"]
            print(f"{'model':<14}{'CV RMSE':>10}{'train RMSE':>12}{'train R2':>10}"
                  f"{'vs baseline':>13}  verdict")
            print("-" * 80)
            for r in sorted(reports, key=lambda r: (r["cv_rmse"] is None,
                                                    r["cv_rmse"] or 0.0)):
                cv = f"{r['cv_rmse']:.5f}" if r["cv_rmse"] is not None else "n/a"
                r2 = f"{r['train_r2']:+.4f}" if r["train_r2"] is not None else "n/a"
                if r["cv_rmse"] is None:
                    ratio, verdict = "n/a", "not cross-validated"
                else:
                    ratio = f"{baseline / r['cv_rmse']:.2f}x"
                    verdict = "USEFUL" if r["beats_baseline"] else "no better than the mean"
                print(f"{r['label']:<14}{cv:>10}{r['train_rmse']:>12.5f}{r2:>10}"
                      f"{ratio:>13}  {verdict}")
            print("-" * 80)
            print(f"mean-baseline RMSE = {baseline:.5f}  (predicting the training mean "
                  f"for every architecture)")
            print("A surrogate is only worth using if its CV RMSE beats that number.")

        print("\n" + "=" * 80)
        bo_evals = min(hybrid.bo_evaluations, len(outcome.history))
        ga_evals = len(outcome.history) - bo_evals
        if abandoned:
            print("SEARCH ABANDONED - the Bayesian phase produced no usable "
                  "population, so the neuroevolution phase never ran.")
            print(f"Reason: {outcome.best_result.error}")
            print("Every performance metric is reported as 0.0 for this run: a "
                  "half-completed hybrid is not comparable with a complete one.")
            print(f"Best fitness: {outcome.best_result}")
            print(f"Evaluations attempted: {len(outcome.history)}/{BUDGET} "
                  f"({bo_evals} Bayesian + {ga_evals} neuroevolution)")
            print(f"Total search time: {outcome.total_search_time:.1f}s")
        else:
            print(f"Best architecture (eval #{outcome.best_evaluation_number} of "
                  f"{len(outcome.history)}): {outcome.best_architecture}")
            print(f"Best fitness: {outcome.best_result}")
            print(f"Time to best: {outcome.time_to_best:.1f}s")
            print(f"Evaluations: {len(outcome.history)}/{BUDGET} "
                  f"({bo_evals} Bayesian + {ga_evals} neuroevolution)")
            print(f"Found in phase: "
                  f"{'Bayesian' if outcome.best_evaluation_number <= bo_evals else 'neuroevolution'}")
            print(f"Total search time: {outcome.total_search_time:.1f}s")
    finally:
        sys.stdout = sys.__stdout__
        log_file.close()
        print(f"Output written to {output_path}")
