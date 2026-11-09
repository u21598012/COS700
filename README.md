# COS700 - A Hybridization of Neuroevolution and Bayesian Optimisation for Neural Network Design

Compares Random Search, Bayesian Optimisation, Neuroevolution and a Bayesian Optimisation + Neuroevolution hybrid for neural architecture search on the MIT-BIH Arrhythmia dataset.

## Introduction

How well a neural network performs depends heavily on its architecture. Designing that architecture by hand, or with exhaustive methods such as grid search, is slow and computationally expensive. Neural Architecture Search (NAS) automates the process, and two common strategies each have a clear weakness:

- **Bayesian Optimisation** guides the search towards promising regions and is sample-efficient. However, it can struggle to explore large and complex search spaces.
- **Neuroevolution** explores the search space well through mutation and crossover. However, every candidate must be fully built and trained, which makes it expensive.

This project combines the two approaches so that each can address the other's limitations:

1. **Bayesian Optimisation phase.** Optuna's TPE sampler runs for the first 40% of the evaluation budget. The architectures it fully evaluates seed the initial population.
2. **Surrogate-assisted Neuroevolution phase.** A genetic algorithm evolves the population. In each generation, an XGBoost surrogate predicts the fitness of every offspring, and only the most promising are actually trained. The surrogate is refitted after every new real result.

The hybrid was compared against standalone Random Search, Bayesian Optimisation and Neuroevolution over 30 matched-seed runs per algorithm.

## Repository Contents

| Path | Contents |
|---|---|
| `main.py` | Entry point for running experiments |
| `configs/experiment.yaml` | Default experiment settings: algorithm, seeds, budget, epochs and hybrid parameters (`algorithm_params`) |
| `configs/search_space.yaml` | The neural architecture search space |
| `data/splits/` | Pre-made train, validation and test splits |
| `results/database/results.db` | Results database |
| `results/csv/` | CSV exports of the results |
| `mlflow.db` | MLflow run tracking |
| `requirements.txt` | Python dependencies |

### Dataset

The experiments use the preprocessed Kaggle version of the MIT-BIH Arrhythmia Database ("ECG Heartbeat Categorization Dataset"). Each row is one segmented heartbeat: 187 signal samples followed by a label for one of five classes (N, S, V, F, Q). The splits are already in `data/splits/`, so there is nothing to download.

| Split | Heartbeats | Use |
|---|---|---|
| Train | 70,043 | Training each candidate network |
| Validation | 17,511 | Fitness (Macro F1) during the search |
| Test | 21,892 | Held out; never seen by the search |

The classes are heavily imbalanced: about 83% of heartbeats are normal. Macro F1 is therefore used as the fitness measure instead of accuracy.

### Experimental Setup

- **Budget:** 50 full evaluations (network trainings) per run, for every algorithm.
- **Training procedure:** fixed for every candidate, so only the architecture varies. Each network is trained for 10 epochs with the Adam optimiser, cross-entropy loss and a batch size of 128.
- **Runs:** 30 per algorithm, using seeds 1–30, paired by seed across all algorithms.
- **Hardware:** a single NVIDIA RTX 5060 Laptop GPU.
- **Statistics:** a Friedman test, followed by Holm-corrected Wilcoxon signed-rank tests, with paired rank-biserial correlation as the effect size.

## Setup

Clone the repo, then from the repo folder:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install optuna==4.9.0 xgboost==3.4.0
```

On Mac or Linux, activate with `source venv/bin/activate`.

To train on an NVIDIA GPU, also run:

```bash
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```

## Check it works

This runs a very small experiment, 2 seeds with 3 networks each:

```bash
python main.py --smoke
```

## Run an experiment

```bash
python main.py
```

This uses the settings in `configs/experiment.yaml`. By default it runs the hybrid on 3 seeds, training 50 networks per seed for 10 epochs each. A single seed can take a few hours on a GPU.

### Choose an algorithm

```bash
python main.py --algorithm random_search
python main.py --algorithm bayesian_optimization
python main.py --algorithm neuroevolution
python main.py --algorithm bayesian_optimization_neuroevolution_hybrid
```

### Other options

```bash
python main.py --seeds 11 22 33
python main.py --budget 20
python main.py --epochs 5
python main.py --device cpu
```

`--budget` is the number of networks trained per seed. `--device` can be `auto`, `cpu` or `cuda`.

Options can be combined:

```bash
python main.py --algorithm neuroevolution --seeds 11 22 --budget 20 --epochs 5
```

To change the defaults, edit `configs/experiment.yaml`. To run all 30 seeds from the study, uncomment the seeds in that file. Hybrid settings are under `algorithm_params` in the same file. The search space is in `configs/search_space.yaml`.

### Viewing results

Results are saved to `results/database/results.db`, with CSV exports in `results/csv/`. MLflow runs are saved to `mlflow.db`. To view them:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Outcomes

### Classification performance (best architecture per run, 30 runs)

| Algorithm | Best Accuracy (mean ± SD) | Best Macro F1 (mean ± SD) |
|---|---|---|
| Random Search | 0.9830 ± 0.0019 | 0.9090 ± 0.0078 |
| Bayesian Optimisation | 0.9841 ± 0.0013 | 0.9173 ± 0.0053 |
| Neuroevolution | 0.9823 ± 0.0021 | 0.9075 ± 0.0103 |
| **Hybrid** | **0.9845 ± 0.0015** | **0.9177 ± 0.0063** |

### Computational cost (hours)

| Algorithm | Total Search Time (median) | Time to Best (median) |
|---|---|---|
| Random Search | 98.17 | 43.76 |
| Bayesian Optimisation | 210.43 | 123.22 |
| Neuroevolution | 73.85 | 55.22 |
| Hybrid | 139.58 | 93.95 |

### Key findings

- **Matches Bayesian Optimisation on classification performance, at lower cost.** The differences in accuracy and Macro F1 between the hybrid and Bayesian Optimisation were not statistically significant. The hybrid's total search time was significantly lower, a reduction of about 34% at the median.
- **Outperforms Neuroevolution and Random Search on classification performance.** It scored significantly higher on both accuracy and Macro F1, but required significantly more search time than either.
- **Broader screening within the same budget.** On average, the surrogate let the hybrid screen 1,322 candidate architectures, against 50 for Random Search and Bayesian Optimisation. Its number of full trainings stayed the same.
- **Intermediate architectural diversity.** The hybrid's best architectures were more varied than those of Bayesian Optimisation, and less varied than those of Neuroevolution and Random Search.
- **No single winner.** The hybrid balances classification performance against search efficiency, rather than outperforming every alternative on every metric.

### Limitations

- The study used a single dataset and a single search space.
- The hybrid was tested with one fixed configuration, including how the budget is split between the two phases.
- Timings are specific to the hardware used.
- Neuroevolution used fewer full evaluations on average (40.8 against 50), which partly explains its lower search time.
