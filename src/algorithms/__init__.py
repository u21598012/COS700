
from src.algorithms.base_algorithm import BaseAlgorithm, SearchOutcome
from src.algorithms.bayesian_optimization import BayesianOptimization
from src.algorithms.bayesian_optimization_neuroevolution_hybrid import (
    BayesianOptimizationNeuroevolutionHybrid,
)
from src.algorithms.hybrid import Hybrid
from src.algorithms.neuroevolution import Neuroevolution
from src.algorithms.random_search import RandomSearch

ALGORITHMS = {
    "random_search": RandomSearch,
    "bayesian_optimization": BayesianOptimization,
    "neuroevolution": Neuroevolution,
    "hybrid": Hybrid,
    
    BayesianOptimizationNeuroevolutionHybrid.name:
        BayesianOptimizationNeuroevolutionHybrid,
}


def get_algorithm(name, search_space, budget, **kwargs):
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
    "BayesianOptimizationNeuroevolutionHybrid",
    "ALGORITHMS",
    "get_algorithm",
]
