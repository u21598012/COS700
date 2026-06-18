# ECG Neural Architecture Search Research Framework Specification

## Purpose

This framework is designed to compare:

* Random Search
* Bayesian Optimization
* Neuroevolution
* Bayesian Optimization + Neuroevolution Hybrid

on the MIT-BIH Arrhythmia dataset using identical:

* Dataset splits
* Search space
* Evaluation procedure
* Logging procedure
* Resource monitoring procedure

The framework must provide complete reproducibility and automatic experiment tracking.

---

# 1. Repository Structure

```text
ecg-hpo-framework/
│
├── .gitignore
├── README.md
├── requirements.txt
├── main.py
│
├── configs/
│   ├── experiment.yaml
│   ├── search_space.yaml
│   ├── dataset.yaml
│   └── tracking.yaml
│
├── data/
│   ├── raw/
│   │   └── mitbih/
│   │
│   ├── processed/
│   │
│   └── splits/
│       ├── train_idx.npy
│       ├── val_idx.npy
│       └── test_idx.npy
│
├── notebooks/
│   ├── setup_validation.ipynb
│   └── experiment_runner.ipynb
│
├── src/
│
│   ├── algorithms/
│   │   ├── base_algorithm.py
│   │   ├── random_search.py
│   │   ├── bayesian_optimization.py
│   │   ├── neuroevolution.py
│   │   └── hybrid.py
│   │
│   ├── data/
│   │   ├── loader.py
│   │   ├── preprocessing.py
│   │   └── split_loader.py
│   │
│   ├── models/
│   │   ├── architecture.py
│   │   ├── architecture_builder.py
│   │   └── search_space.py
│   │
│   ├── evaluation/
│   │   ├── evaluator.py
│   │   ├── metrics.py
│   │   ├── profiler.py
│   │   └── timer.py
│   │
│   ├── tracking/
│   │   ├── database.py
│   │   ├── mlflow_logger.py
│   │   └── artifact_manager.py
│   │
│   ├── experiment/
│   │   ├── runner.py
│   │   └── experiment_manager.py
│   │
│   └── utils/
│       ├── config.py
│       ├── seed.py
│       ├── paths.py
│       └── system_info.py
│
├── tests/
│   ├── test_database.py
│   ├── test_mlflow.py
│   ├── test_search_space.py
│   └── test_runner.py
│
├── results/
│   │
│   ├── database/
│   │   └── results.db
│   │
│   ├── csv/
│   │   ├── runs.csv
│   │   └── evaluations.csv
│   │
│   ├── models/
│   │
│   ├── plots/
│   │
│   └── logs/
│
└── mlruns/
```

---

# 2. Framework Execution Sequence

Every algorithm must follow this sequence.

```text
Load Configuration
        ↓
Load Dataset
        ↓
Load Fixed Splits
        ↓
Initialize Database
        ↓
Initialize MLflow
        ↓
For Each Seed
        ↓
Set Seed
        ↓
Create Run Record
        ↓
Execute Search Algorithm
        ↓
Generate Candidate
        ↓
Build Model
        ↓
Train Model
        ↓
Validate Model
        ↓
Compute Metrics
        ↓
Log Results
        ↓
Update Best Candidate
        ↓
Repeat Until Budget Exhausted
        ↓
Save Best Model
        ↓
Export Results
        ↓
Close Run
```

---

# 3. Phase 1 - Project Setup

## Create Virtual Environment

```bash
python -m venv .venv
```

Activate:

```bash
.venv\Scripts\activate
```

Install packages:

```bash
pip install torch torchvision torchaudio

pip install numpy pandas scipy

pip install scikit-learn

pip install matplotlib

pip install seaborn

pip install mlflow

pip install pyyaml

pip install psutil

pip install sqlalchemy

pip install jupyter

pip install tqdm
```

Create requirements:

```bash
pip freeze > requirements.txt
```

---

# 4. Phase 2 - Create Directory Structure

Create every directory exactly as specified.

Do not skip any.

The framework assumes these paths exist.

---

# 5. Phase 3 - Configuration System

## experiment.yaml

Contains:

* algorithms
* seeds
* budget
* training settings
* tracking settings

## search_space.yaml

Contains:

* global parameters
* layer parameters

## dataset.yaml

Contains:

* dataset paths
* split paths

## tracking.yaml

Contains:

* mlflow settings
* database settings
* export settings

---

# 6. Phase 4 - Fixed Dataset Splits

Generate train/validation/test splits once.

Save:

```text
train_idx.npy
val_idx.npy
test_idx.npy
```

inside:

```text
data/splits/
```

Commit these files.

Never regenerate.

Every algorithm uses the same files.

---

# 7. Phase 5 - Database Setup

Database:

```text
results/database/results.db
```

Tables:

## experiments

Stores experiment metadata.

## runs

One row per seed.

## evaluations

One row per architecture evaluated.

## best_models

Stores best architecture per run.

Database initialization must execute before any experiment begins.

---

# 8. Phase 6 - MLflow Setup

Tracking URI:

```python
mlflow.set_tracking_uri("file:./mlruns")
```

Experiment:

```python
mlflow.set_experiment(
    experiment_name
)
```

Every seed becomes a separate MLflow run.

---

# 9. Phase 7 - Search Space System

Search space loader must:

1. Load YAML
2. Validate ranges
3. Sample parameters
4. Return ArchitectureConfig

The search algorithm never reads YAML directly.

Only the search space module does.

---

# 10. Phase 8 - Architecture Representation

Every candidate architecture must use the same schema.

```python
ArchitectureConfig
```

containing:

```text
learning_rate
num_layers
layers[]
```

Each layer contains:

```text
layer_type
hidden_size
dropout
activation
kernel_size
bidirectional
```

All algorithms must generate this format.

---

# 11. Phase 9 - Model Builder

Input:

```python
ArchitectureConfig
```

Output:

```python
torch.nn.Module
```

The builder converts architecture descriptions into executable models.

Search algorithms never create PyTorch layers directly.

---

# 12. Phase 10 - Evaluator

Input:

```python
architecture
train_loader
val_loader
config
```

Output:

```python
EvaluationResult
```

Metrics:

* Accuracy
* Precision
* Recall
* Macro F1
* Training Time
* Inference Time
* Peak Memory

Every algorithm must use the same evaluator.

---

# 13. Phase 11 - Resource Monitoring

Record:

## CPU

```text
Average CPU %
Peak CPU %
```

## RAM

```text
Average RAM MB
Peak RAM MB
```

## GPU

```text
Peak GPU Memory MB
```

Sampling begins before training.

Sampling ends after evaluation.

---

# 14. Phase 12 - Artifact Storage

For each run:

```text
models/
best_model.pt

architectures/
candidate_001.json

predictions/
predictions.csv

plots/
confusion_matrix.png

search_progress.csv
```

Artifacts must be saved locally and logged to MLflow.

---

# 15. Phase 13 - CSV Export

## runs.csv

One row per run.

Columns:

```text
algorithm
seed
best_accuracy
best_precision
best_recall
best_macro_f1
total_search_time
time_to_best
peak_memory_mb
best_architecture
```

## evaluations.csv

One row per evaluated candidate.

Columns:

```text
algorithm
seed
evaluation_number
architecture_json
accuracy
precision
recall
macro_f1
training_time
peak_memory_mb
```

---

# 16. Initial Dry Run Procedure

Before implementing Random Search:

## Step 1

Run:

```bash
python main.py
```

## Step 2

Framework loads:

```text
experiment.yaml
search_space.yaml
```

## Step 3

Framework initializes:

```text
results.db
mlruns/
```

## Step 4

Framework creates:

```text
Run:
Seed 11

Run:
Seed 22

Run:
Seed 33
```

## Step 5

Framework generates dummy candidate architectures.

No model training.

No dataset required.

Example:

```json
{
  "learning_rate": 0.001,
  "num_layers": 2
}
```

## Step 6

Framework logs:

* Seed
* Candidate
* Timing
* Memory

to:

```text
SQLite
MLflow
```

## Step 7

Verify:

### Database

```sql
SELECT * FROM runs;
SELECT * FROM evaluations;
```

contain rows.

### MLflow

Open:

```bash
mlflow ui
```

Visit:

http://localhost:5000

Verify:

* Experiment exists
* Runs exist
* Metrics exist

## Step 8

Export:

```text
runs.csv
evaluations.csv
```

Verify files are created.

Only after this succeeds should model generation and Random Search implementation begin.

---

# 17. Random Search Integration

Once dry run succeeds:

Replace:

```python
generate_dummy_candidate()
```

with:

```python
sample_architecture()
```

Replace:

```python
dummy_evaluation()
```

with:

```python
evaluate_architecture()
```

Everything else remains unchanged.

This guarantees that BO, Neuroevolution and Hybrid can later reuse the identical infrastructure.
