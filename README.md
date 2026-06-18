# COS700 — ECG Neural Architecture Search Framework

A reproducible framework for comparing **Random Search**, **Bayesian Optimization**,
**Neuroevolution**, and a **BO + Neuroevolution Hybrid** for neural architecture
search on the MIT-BIH Arrhythmia dataset. All algorithms share identical dataset
splits, search space, evaluator, logging, and resource monitoring.

See [experimentation_framework.md](experimentation_framework.md) for the full spec.

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Place the MIT-BIH CSVs at `data/raw/mitbih_train.csv` and `data/raw/mitbih_test.csv`.

**Fixed splits:** the framework only *loads* splits — it never creates them. Create
`data/splits/{train,val,test}_idx.npy` once yourself first (train_idx/val_idx index
`mitbih_train.csv` rows; test_idx indexes `mitbih_test.csv` rows). They are then
reused unchanged by every run.

## Quickstart

```bash
# tiny end-to-end sanity run (a few seconds): trains, logs, writes the DB + CSVs
# (requires the split files to already exist)
python main.py --smoke

# full experiment defined in configs/experiment.yaml
python main.py
```

Or open [notebooks/experiment_runner.ipynb](notebooks/experiment_runner.ipynb) and
run the cells. Use [notebooks/setup_validation.ipynb](notebooks/setup_validation.ipynb)
to verify the environment first.

## Where results land

| Output | Location |
| --- | --- |
| SQLite results DB | `results/database/results.db` (tables: `experiments`, `runs`, `evaluations`, `best_models`) |
| Run/eval CSVs | `results/csv/runs.csv`, `results/csv/evaluations.csv` |
| Best models / architectures | `results/models/`, `results/architectures/` |
| MLflow runs | `mlflow.db` (view with `mlflow ui --backend-store-uri sqlite:///mlflow.db`) |

## Configuration (`configs/`)

| File | Controls |
| --- | --- |
| `experiment.yaml` | algorithm, seeds, budget, training settings, smoke knobs |
| `search_space.yaml` | global + per-layer search ranges |
| `dataset.yaml` | dataset paths, splits, preprocessing, dataloader |
| `tracking.yaml` | MLflow / database / artifact / export settings |

## Adding your model architectures

The infrastructure is complete and runnable. The **only** thing to implement is how
an architecture spec becomes real PyTorch layers, in
[`src/models/architecture_builder.py`](src/models/architecture_builder.py)
(`build_model`). A placeholder MLP ships so the pipeline runs today; replace it with
your CNN/LSTM/GRU construction and nothing else needs to change.

## Project layout

```
configs/      YAML configuration
src/
  algorithms/   search strategies (random_search implemented; others stubbed)
  data/         loading, preprocessing, fixed splits, dataloaders
  models/       architecture schema, search space, model builder (your layers)
  evaluation/   evaluator, metrics, profiler (resource monitor), timer
  tracking/     SQLite DB, MLflow logger, artifacts, CSV export
  experiment/   per-seed Runner + ExperimentManager orchestration
  utils/        config, paths, seeding, system info
notebooks/    setup_validation.ipynb, experiment_runner.ipynb
results/      DB, CSVs, models, plots, logs (generated)
```
