"""Search-loop tests for RandomSearch (no data / training needed).

Drives the algorithm with a fake ``evaluate_fn`` so we can assert the shared
best-tracking / history / budget logic in BaseAlgorithm works.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.algorithms import get_algorithm
from src.evaluation.evaluator import EvaluationResult
from src.models.search_space import SearchSpace

SPACE = {
    "global": {"learning_rate": {"min": 0.001, "max": 0.01}, "num_layers": {"min": 1, "max": 2}},
    "layer": {"type": {"choices": ["CNN"]}, "hidden_size": {"min": 8, "max": 16},
              "dropout": {"min": 0.0, "max": 0.1}},
}


def test_random_search_respects_budget_and_picks_best():
    space = SearchSpace(SPACE)
    budget = 6
    algo = get_algorithm("random_search", space, budget, selection_metric="macro_f1")

    scores = [0.1, 0.5, 0.3, 0.9, 0.2, 0.4]  # best is index 3 -> eval number 4
    calls = {"n": 0}
    logged = []

    def evaluate_fn(arch):
        f1 = scores[calls["n"]]
        calls["n"] += 1
        return EvaluationResult(accuracy=f1, macro_f1=f1)

    def on_evaluation(num, arch, result):
        logged.append(num)

    outcome = algo.search(evaluate_fn, on_evaluation)

    assert calls["n"] == budget
    assert len(outcome.history) == budget
    assert logged == list(range(1, budget + 1))
    assert outcome.best_evaluation_number == 4
    assert abs(outcome.best_result.macro_f1 - 0.9) < 1e-9
    assert outcome.total_search_time >= 0.0


def test_failed_evaluations_are_skipped_for_best():
    space = SearchSpace(SPACE)
    algo = get_algorithm("random_search", space, 3)

    def evaluate_fn(arch):
        return EvaluationResult(failed=True, error="boom")

    outcome = algo.search(evaluate_fn)
    assert outcome.best_result is None
    assert len(outcome.history) == 3


if __name__ == "__main__":
    test_random_search_respects_budget_and_picks_best()
    test_failed_evaluations_are_skipped_for_best()
    print("test_runner: OK")
