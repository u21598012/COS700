

from src.algorithms.base_algorithm import BaseAlgorithm


class Hybrid(BaseAlgorithm):
    name = "hybrid"

    def search(self, evaluate_fn, on_evaluation=None):
        raise NotImplementedError(
            "Hybrid.search is not implemented yet. Use random_search for now."
        )
