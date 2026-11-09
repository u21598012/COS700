# Framework Execution Sequence

A step-by-step walkthrough of what each component does and in what order.


---

## Phase 1 — Entry Point (`main.py`)

Parse CLI args (`--algorithm`, `--seeds`, `--budget`, etc.), load all YAML configs, apply any overrides, then hand off to `ExperimentManager`.

---

## Phase 2 — Setup (`src/experiment/experiment_manager.py`)

1. **Create directories** — output folders, artifact paths
2. **Initialise SQLite DB** — blank schema ready to receive records
3. **Initialise MLflow** — sets tracking URI and experiment name
4. **Verify fixed splits exist** — `data/splits/{train,val,test}_idx.npy` must already exist; the framework never creates them, only loads them

---

## Phase 3 — Load Data (`src/data/split_loader.py`)

Load the `.npy` index files, slice the preprocessed CSVs into train/val/test arrays (already z-score normalised), optionally subsample for smoke runs, then wrap in PyTorch `DataLoader`s. Returns loaders + metadata (`input_dim`, `num_classes`).

---

## Phase 4 — Build Search Space (`src/models/search_space.py`)

Parse `search_space.yaml` and validate all ranges (learning rate, num layers, hidden size, dropout, layer types). The `SearchSpace` object is the only thing that reads this config; algorithms ask it for samples.

---

## Phase 5 — Register Experiment in DB

Insert one experiment record (algorithm name, budget, config snapshot, system info) with a UUID. All per-seed runs link back to this.

---

## Phase 6 — Seed Loop (`src/experiment/runner.py`)

For **each seed** in config:

**6a. Set seed** — `set_seed(seed)` fixes Python, NumPy, and PyTorch RNG for reproducibility.

**6b. Instantiate algorithm** — one of: `random_search`, `bayesian_optimization`, `neuroevolution`, `hybrid`. All share the same `BaseAlgorithm` interface.

**6c. Open MLflow run** — creates a new run context; params (algorithm, seed, budget, epochs) are logged immediately.

**6d. Create DB run record** — inserts a row linking this seed run to the experiment and MLflow run ID.

---

## Phase 7 — Search Loop (`src/algorithms/base_algorithm.py`)

The algorithm calls `_run_loop`, which iterates up to `budget` evaluations. For each candidate architecture:

1. **Sample architecture** — `SearchSpace.sample_architecture()` draws random layer types, sizes, dropout, learning rate
2. **Evaluate** — calls `evaluate_fn(architecture)`
3. **Track best** — compares score on `selection_metric` (default: `macro_f1`)
4. **Callback** — fires `on_evaluation(eval_number, architecture, result)` which logs to DB + MLflow

> Bayesian, Neuroevolution, and Hybrid override the sampling step; evaluation and logging are always identical across all algorithms.

---

## Phase 8 — Evaluate Architecture (`src/evaluation/evaluator.py`)

For each candidate:

1. **Build model** from the architecture spec (`build_model`)
2. **Train** for N epochs with Adam + CrossEntropyLoss
3. **Validate** on val split — collect predictions
4. **Compute metrics** — accuracy, precision, recall, macro F1
5. **Record resource usage** — CPU %, RAM MB, GPU MB (via `ResourceMonitor`)

Returns a uniform `EvaluationResult`. Failures are caught and marked `failed=True` so the search continues without crashing.

---

## Phase 9 — Per-Evaluation Logging

After each evaluation:

- Insert evaluation record into SQLite (architecture JSON + all metrics)
- Save architecture JSON as artifact
- Log metrics to MLflow with `step=eval_number`

---

## Phase 10 — Finalise Seed Run

After the budget is exhausted:

- Save search progress history (score curve over evaluations)
- Save best architecture JSON + `.pt` model file
- Save predictions array + confusion matrix plot
- Finalise run record in SQLite (status, best metrics, timing)
- Log summary metrics to MLflow (`best_accuracy`, `best_macro_f1`, `total_search_time`, `time_to_best`)
- Return a summary dict for this seed

---

## Phase 11 — Export & Report

After all seeds complete:

- Export `runs.csv` and `evaluations.csv` from SQLite
- Print per-seed summary table to stdout (best F1, best accuracy, total time)

---

## Summary Flow

```
main.py
  └─ ExperimentManager.run()
       ├─ setup()           → dirs, DB, MLflow, verify splits
       ├─ load_data()       → DataLoaders (train/val/test)
       ├─ SearchSpace()     → validated sampling object
       └─ for each seed:
            Runner.run_seed()
              ├─ set_seed()
              ├─ Algorithm.search()
              │    └─ for each eval:
              │         ├─ sample_architecture()
              │         ├─ evaluate_architecture()  → train + val + metrics
              │         └─ log to DB + MLflow
              └─ finalize()  → save best model, confusion matrix, CSV export
```
