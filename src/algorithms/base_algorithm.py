
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional


@dataclass
class SearchOutcome:

    best_architecture: Optional[object] = None
    best_result: Optional[object] = None
    best_evaluation_number: Optional[int] = None
    total_search_time: float = 0.0
    time_to_best: float = 0.0
    history: List[dict] = field(default_factory=list)


class BaseAlgorithm:
    
    name = "base"

    def __init__(self, search_space, budget, selection_metric="macro_f1"):
        self.search_space = search_space
        self.budget = budget
        self.selection_metric = selection_metric

    def set_run_context(self, run_id, seed):
       
        return self

    def _score(self, result):
        if result is None or getattr(result, "failed", False):
            return float("-inf")
        return getattr(result, self.selection_metric, float("-inf"))

    def search(self, evaluate_fn: Callable, on_evaluation: Optional[Callable] = None) -> SearchOutcome:
       
        raise NotImplementedError

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
