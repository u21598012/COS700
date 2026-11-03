# Running an Experiment

---

## Prerequisites

- Python virtual environment activated (`venv/Scripts/activate` on Windows)
- Raw data files present at `data/raw/mitbih_train.csv` and `data/raw/mitbih_test.csv`
- Fixed splits generated and saved at `data/splits/train_idx.npy`, `val_idx.npy`, `test_idx.npy`

> The framework never generates splits itself. If they don't exist yet, run the data preparation notebook (`notebooks/setup_validation.ipynb`) once to create them.

---

## Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

---

## Step 2 — Configure the experiment

Edit the four YAML files in `configs/` as needed before running:

| File | What to change |
|---|---|
| `configs/experiment.yaml` | `algorithm`, `seeds`, `budget.evaluations`, `training.epochs`, `experiment_name` |
| `configs/search_space.yaml` | Layer type choices, hidden size range, dropout range, learning rate range |
| `configs/dataset.yaml` | `num_classes`, file paths, batch size |
| `configs/tracking.yaml` | MLflow tracking URI (defaults to local `mlflow.db`) |

The default config runs **random search** for **100 evaluations** across **30 seeds**, training each candidate for **10 epochs**.

---

## Step 3 — Run the experiment

**Default (uses `configs/experiment.yaml` as-is):**

```bash
python main.py
```

**Smoke test — fast end-to-end sanity check (finishes in seconds):**

```bash
python main.py --smoke
```

**Override algorithm and budget at the command line:**

```bash
python main.py --algorithm bayesian_optimization --budget 50 --seeds 42 123
```

**Force CPU (useful if CUDA is available but you want a CPU run):**

```bash
python main.py --device cpu
```

Available algorithms: `random_search`, `bayesian_optimization`, `neuroevolution`, `hybrid`

---

## Step 4 — Monitor results

While running, the console prints a live line per evaluation:

```
=== Run 1/30 | random_search | seed 5912847 | device cpu ===
  [seed 5912847] eval 1/100  f1=0.8231 acc=0.8410
  [seed 5912847] eval 2/100  f1=0.8654 acc=0.8791
  ...
```

When all seeds finish, a summary table is printed:

```
Done. Run summaries:
  seed 5912847: best_f1=0.9102 best_acc=0.9244 time=183.4s
  seed 1384225: best_f1=0.9057 best_acc=0.9198 time=177.1s
  ...
```

---

## Step 5 — Inspect outputs

| Output | Location |
|---|---|
| SQLite database | `mlflow.db` (or path set in `configs/tracking.yaml`) |
| MLflow UI | Run `mlflow ui` then open `http://localhost:5000` |
| Per-run CSV exports | `results/runs.csv`, `results/evaluations.csv` |
| Best model weights | `results/artifacts/<run_id>_best.pt` |
| Best architecture JSON | `results/artifacts/<run_id>_best.json` |
| Confusion matrix | `results/artifacts/<run_id>_best_confusion_matrix.png` |
| Search progress curve | `results/artifacts/<run_id>_search_progress.json` |
