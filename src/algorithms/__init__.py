"""Search algorithm registry.

Use :func:`get_algorithm` so the experiment runner can pick an algorithm by the
string name in ``experiment.yaml`` (``algorithm:``).
"""

from src.algorithms.base_algorithm import BaseAlgorithm, SearchOutcome
from src.algorithms.bayesian_optimization import BayesianOptimization
from src.algorithms.hybrid import Hybrid
from src.algorithms.neuroevolution import Neuroevolution
from src.algorithms.random_search import RandomSearch

ALGORITHMS = {
    "random_search": RandomSearch,
    "bayesian_optimization": BayesianOptimization,
    "neuroevolution": Neuroevolution,
    "hybrid": Hybrid,
}


def get_algorithm(name, search_space, budget, **kwargs):
    """Instantiate an algorithm by name."""
    if name not in ALGORITHMS:
        raise ValueError(
            f"Unknown algorithm '{name}'. Available: {sorted(ALGORITHMS)}"
        )
    return ALGORITHMS[name](search_space, budget, **kwargs)


__all__ = [
    "BaseAlgorithm",
    "SearchOutcome",
    "RandomSearch",
    "BayesianOptimization",
    "Neuroevolution",
    "Hybrid",
    "ALGORITHMS",
    "get_algorithm",
]
