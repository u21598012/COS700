"""Command-line entry point for the ECG NAS framework.

Examples
--------
    python main.py                 # run the experiment defined in configs/
    python main.py --smoke         # tiny, fast end-to-end run (sanity check)
    python main.py --algorithm random_search --seeds 11 22 --budget 5
"""

import argparse

from src.experiment.experiment_manager import ExperimentManager
from src.utils.config import apply_overrides, load_configs


# Tiny overrides so the whole pipeline finishes in seconds end-to-end.
SMOKE_OVERRIDES = {
    "experiment": {
        "seeds": [11, 22],
        "budget": {"evaluations": 3},
        "training": {"epochs": 1, "batch_size": 128},
        "data": {"max_train_samples": 2000, "max_val_samples": 1000, "max_test_samples": 1000},
    }
}


def build_overrides(args):
    overrides = {"experiment": {}}
    if args.algorithm:
        overrides["experiment"]["algorithm"] = args.algorithm
    if args.seeds:
        overrides["experiment"]["seeds"] = args.seeds
    if args.budget:
        overrides["experiment"]["budget"] = {"evaluations": args.budget}
    if args.epochs:
        overrides["experiment"]["training"] = {"epochs": args.epochs}
    return overrides


def main():
    parser = argparse.ArgumentParser(description="ECG NAS experiment runner")
    parser.add_argument("--smoke", action="store_true", help="tiny fast run for sanity checks")
    parser.add_argument("--algorithm", type=str, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                        help="auto (GPU if available) / cpu / cuda")
    args = parser.parse_args()

    configs = load_configs()

    if args.smoke:
        configs = apply_overrides(configs, SMOKE_OVERRIDES)
    configs = apply_overrides(configs, build_overrides(args))

    manager = ExperimentManager(configs=configs, device=args.device)
    result = manager.run()

    print("\nDone. Run summaries:")
    for s in result["summaries"]:
        print(
            f"  seed {s['seed']}: best_f1={s['best_macro_f1']:.4f} "
            f"best_acc={s['best_accuracy']:.4f} "
            f"time={s['total_search_time']:.1f}s"
        )


if __name__ == "__main__":
    main()
