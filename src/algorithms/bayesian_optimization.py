"""Bayesian Optimization search (placeholder).

Implement the BO loop here later. Subclass :class:`BaseAlgorithm` and override
``search`` (or feed candidates into ``self._run_loop``) so it reuses the same
evaluation + logging infrastructure as Random Search.
"""

from src.algorithms.base_algorithm import BaseAlgorithm


class BayesianOptimization(BaseAlgorithm):
    name = "bayesian_optimization"

    def search(self, evaluate_fn, on_evaluation=None):
        raise NotImplementedError(
            "BayesianOptimization.search is not implemented yet. "
            "Use random_search for now."
        )
