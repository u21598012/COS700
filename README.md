# COS700

Compares Random Search, Bayesian Optimization, Neuroevolution and a BO + Neuroevolution hybrid for neural architecture search on the MIT-BIH Arrhythmia dataset.

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

The data splits are already in `data/splits/`, so there is nothing to download.

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

## Choose an algorithm

```bash
python main.py --algorithm random_search
python main.py --algorithm bayesian_optimization
python main.py --algorithm neuroevolution
python main.py --algorithm bayesian_optimization_neuroevolution_hybrid
```

## Other options

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

## Results

Results are saved to `results/database/results.db`, with CSV exports in `results/csv/`. MLflow runs are saved to `mlflow.db`. To view them:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
