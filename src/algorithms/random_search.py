"""Random Search (the first algorithm; framework spec, section 17).

Samples ``budget`` architectures uniformly from the search space and keeps the
best. Because all the heavy lifting (build/train/evaluate/log) is injected, the
later algorithms (BO, Neuroevolution, Hybrid) reuse this exact machinery.
"""

from src.algorithms.base_algorithm import BaseAlgorithm


class RandomSearch(BaseAlgorithm):
    name = "random_search"

    def search(self, evaluate_fn, on_evaluation=None):
        candidates = (self.search_space.sample_architecture() for _ in range(self.budget))
        return self._run_loop(candidates, evaluate_fn, on_evaluation)
