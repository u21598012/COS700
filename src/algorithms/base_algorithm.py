"""Base class every search algorithm implements.

The contract is intentionally tiny so Random Search, Bayesian Optimization,
Neuroevolution and the Hybrid all share the identical evaluation + logging
infrastructure (framework spec, section 17). An algorithm only decides *which*
architectures to try; building/training/logging is handled elsewhere and passed
in as callbacks.
"""

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional


@dataclass
class SearchOutcome:
    """Everything a run needs to know after a search finishes."""

    best_architecture: Optional[object] = None
    best_result: Optional[object] = None
    best_evaluation_number: Optional[int] = None
    total_search_time: float = 0.0
    time_to_best: float = 0.0
    history: List[dict] = field(default_factory=list)


class BaseAlgorithm:
    """Subclasses implement :meth:`search`.

    Parameters
    ----------
    search_space : SearchSpace
        Produces candidate architectures / bounds.
    budget : int
        Number of architectures to evaluate.
    selection_metric : str
        Field of EvaluationResult used to pick "best" (default macro_f1).
    """

    name = "base"

    def __init__(self, search_space, budget, selection_metric="macro_f1"):
        self.search_space = search_space
        self.budget = budget
        self.selection_metric = selection_metric

    def _score(self, result):
        if result is None or getattr(result, "failed", False):
            return float("-inf")
        return getattr(result, self.selection_metric, float("-inf"))

    def search(self, evaluate_fn: Callable, on_evaluation: Optional[Callable] = None) -> SearchOutcome:
        """Run the search loop.

        ``evaluate_fn(architecture) -> EvaluationResult``
        ``on_evaluation(eval_number, architecture, result)`` is called after each
        evaluation (used by the Runner to log to the DB + MLflow).
        """
        raise NotImplementedError

    # Shared helper so subclasses don't duplicate best-tracking/timing logic.
    def _run_loop(self, candidate_iter, evaluate_fn, on_evaluation):
        outcome = SearchOutcome()
        best_score = float("-inf")
        start = time.perf_counter()

        for eval_number, architecture in enumerate(candidate_iter, start=1):
            result = evaluate_fn(architecture)
            elapsed = time.perf_counter() - start

            score = self._score(result)
            if score > best_score:
                best_score = score
                outcome.best_architecture = architecture
                outcome.best_result = result
                outcome.best_evaluation_number = eval_number
                outcome.time_to_best = elapsed

            outcome.history.append({
                "evaluation_number": eval_number,
                "score": score,
                "elapsed": elapsed,
            })

            if on_evaluation is not None:
                on_evaluation(eval_number, architecture, result)

        outcome.total_search_time = time.perf_counter() - start
        return outcome
