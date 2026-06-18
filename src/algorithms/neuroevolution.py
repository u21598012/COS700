"""Neuroevolution search (placeholder).

Implement the evolutionary loop here later (population, mutation/crossover over
ArchitectureConfig, selection). Reuse ``self._run_loop`` / the injected
``evaluate_fn`` so logging stays identical across algorithms.
"""

from src.algorithms.base_algorithm import BaseAlgorithm


class Neuroevolution(BaseAlgorithm):
    name = "neuroevolution"

    def search(self, evaluate_fn, on_evaluation=None):
        raise NotImplementedError(
            "Neuroevolution.search is not implemented yet. "
            "Use random_search for now."
        )
