"""End-to-end unit tests for the Neuroevolution genetic algorithm.

Goal (per request): exercise EVERY GA component -- generation, evaluation,
selection, mutation, crossover -- first in isolation and then together, over a
range of cases from empty/minimal architectures to maximal ones, confirming
error handling and that every candidate the GA can produce is *valid*.

Design rules:
  * NOTHING is mocked. We use the real ``SearchSpace``, real
    ``ArchitectureConfig``/``LayerConfig``, real ``EvaluationResult`` objects,
    and validate candidates by actually calling ``build_model`` and running a
    forward pass (the hard definition of "valid" in this framework).
  * "Fitness" is a real, deterministic function that returns a real
    ``EvaluationResult`` -- not a Mock -- so selection behaviour is assertable.

Run it either way::

    ./venv/Scripts/python.exe -m pytest tests/test_genetic_algorithm.py -v
    ./venv/Scripts/python.exe -m tests.test_genetic_algorithm      # no pytest needed
"""

import random
import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.algorithms.neuroevolution import Neuroevolution
from src.evaluation.evaluator import EvaluationResult
from src.models.architecture import ArchitectureConfig, LayerConfig
from src.models.architecture_builder import build_model
from src.models.search_space import SearchSpace

# ---------------------------------------------------------------------------
# Fixtures / constants
# ---------------------------------------------------------------------------
INPUT_DIM = 187          # raw ECG beat length (MIT-BIH)
NUM_CLASSES = 5
SEED = 1234

# A real search space (matches configs/search_space.yaml shape) used for the
# "declared bounds" validity checks. Kept explicit so the tests don't depend on
# the YAML on disk changing under them.
SPACE_DICT = {
    "global": {
        "learning_rate": {"min": 0.0001, "max": 0.01},
        "num_layers": {"min": 1, "max": 5},
    },
    "layer": {
        "type": {"choices": ["CNN", "LSTM", "GRU"]},
        "hidden_size": {"min": 16, "max": 256},
        "dropout": {"min": 0.0, "max": 0.5},
        "activation": {"choices": ["relu", "tanh", "elu"]},
        "kernel_size": {"choices": [3, 5, 7]},
        "bidirectional": {"choices": [True, False]},
        "pooling": {"choices": [None, "max", "avg"]},
    },
}


def make_space():
    return SearchSpace(SPACE_DICT)


def make_algo(budget=6, space=None):
    """Fresh algorithm with per-instance state.

    ``population_metadata`` / ``best_architecture`` are declared as *class*
    attributes in Neuroevolution, so we shadow them with instance attributes to
    keep every test isolated (see test_class_attribute_state_is_not_shared).
    """
    algo = Neuroevolution(space or make_space(), budget=budget)
    algo.population = []
    algo.population_metadata = {}
    algo.best_architecture = None
    return algo


def make_layer(layer_type="CNN", hidden_size=32, dropout=0.1, activation="relu"):
    """A hand-built, in-space LayerConfig with the right type-specific fields."""
    return LayerConfig(
        layer_type=layer_type,
        hidden_size=hidden_size,
        dropout=dropout,
        activation=activation,
        kernel_size=3 if layer_type == "CNN" else None,
        bidirectional=False if layer_type in ("LSTM", "GRU") else None,
        pooling=None,
    )


def arch_with_layers(n, layer_type="CNN", lr=0.005):
    layers = [make_layer(layer_type) for _ in range(n)]
    return ArchitectureConfig(learning_rate=lr, num_layers=n, layers=layers)


def real_fitness(architecture):
    """A genuine (non-mock) evaluator: deterministic fitness from structure.

    Returns a real EvaluationResult so downstream code sees exactly what the
    training evaluator would hand it, minus the actual training.
    """
    f1 = min(1.0, 0.05 * architecture.num_layers + 10.0 * architecture.learning_rate)
    return EvaluationResult(accuracy=f1, macro_f1=f1)


def seed_population(algo, archs_and_f1):
    """Populate population + population_metadata directly with real results."""
    algo.population = [arch for arch, _ in archs_and_f1]
    algo.population_metadata = {}
    for arch, f1 in archs_and_f1:
        algo.population_metadata[id(arch)] = {
            "id": id(arch),
            "architecture": arch,
            "fitness": EvaluationResult(accuracy=f1, macro_f1=f1),
        }


@contextmanager
def raises(exc_types):
    """pytest.raises-free assertion so this file runs under plain python too."""
    try:
        yield
    except exc_types:
        return
    else:
        raise AssertionError(f"expected {exc_types} to be raised, but nothing was")


# ---------------------------------------------------------------------------
# Validity helpers -- the definition of a "valid candidate"
# ---------------------------------------------------------------------------
def assert_structural(arch):
    """Internal consistency that every candidate must keep, always."""
    assert isinstance(arch, ArchitectureConfig)
    assert arch.num_layers == len(arch.layers), (
        f"num_layers={arch.num_layers} but len(layers)={len(arch.layers)}"
    )
    for layer in arch.layers:
        assert isinstance(layer, LayerConfig)


def assert_buildable(arch):
    """Hard validity: the candidate builds and runs a forward pass end-to-end."""
    model = build_model(arch, INPUT_DIM, NUM_CLASSES)
    out = model(torch.randn(2, INPUT_DIM))
    assert out.shape == (2, NUM_CLASSES), out.shape


def assert_layer_in_space(layer, space=None):
    """Declared-bounds validity for a single layer."""
    space = space or make_space()
    L = space.space["layer"]
    assert layer.layer_type in L["type"]["choices"], layer.layer_type
    assert L["hidden_size"]["min"] <= layer.hidden_size <= L["hidden_size"]["max"], (
        f"hidden_size {layer.hidden_size} out of declared range"
    )
    assert L["dropout"]["min"] <= layer.dropout <= L["dropout"]["max"], (
        f"dropout {layer.dropout} out of declared range"
    )
    if layer.activation is not None:
        assert layer.activation in L["activation"]["choices"], layer.activation


def assert_in_search_space(arch, space=None):
    """Declared-bounds validity: every field lies within the search space."""
    space = space or make_space()
    g = space.space["global"]
    assert g["learning_rate"]["min"] <= arch.learning_rate <= g["learning_rate"]["max"], (
        f"learning_rate {arch.learning_rate} out of declared range"
    )
    assert g["num_layers"]["min"] <= arch.num_layers <= g["num_layers"]["max"], (
        f"num_layers {arch.num_layers} out of declared range"
    )
    for layer in arch.layers:
        assert_layer_in_space(layer, space)


# ===========================================================================
# 1. GENERATION  (generate_population)
# ===========================================================================
def test_generate_population_size_matches_request():
    for size in (1, 5, 20):
        algo = make_algo(budget=size)
        algo.generate_population(size)
        pop = list(algo.population)
        assert len(pop) == size


def test_generate_population_zero_is_empty():
    algo = make_algo()
    algo.generate_population(0)
    assert list(algo.population) == []


def test_generate_population_negative_is_empty_not_crash():
    algo = make_algo()
    algo.generate_population(-3)          # range(-3) -> empty, must not raise
    assert list(algo.population) == []


def test_generated_candidates_are_valid_and_in_space():
    random.seed(SEED)
    algo = make_algo()
    algo.generate_population(30)
    for arch in algo.population:
        assert_structural(arch)
        assert_in_search_space(arch)
        assert_buildable(arch)


def test_population_is_reusable_list():
    """population is a list, so it can be iterated repeatedly -- steady-state
    replacement appends/removes from it, which a generator can't support."""
    algo = make_algo()
    algo.generate_population(4)
    first = list(algo.population)
    second = list(algo.population)
    assert len(first) == 4
    assert first == second       # re-iterable, not exhausted after one pass


# ===========================================================================
# 2. EVALUATION  (evaluate_architecture / evaluate_population)
# ===========================================================================
def test_evaluate_architecture_returns_result_and_calls_callback():
    algo = make_algo()
    arch = arch_with_layers(2)
    logged = []
    result = algo.evaluate_architecture(
        arch, real_fitness, lambda n, a, r: logged.append((n, a, r))
    )
    assert isinstance(result, EvaluationResult)
    assert len(logged) == 1
    assert logged[0][1] is arch


def test_evaluate_architecture_without_callback():
    algo = make_algo()
    result = algo.evaluate_architecture(arch_with_layers(1), real_fitness, None)
    assert isinstance(result, EvaluationResult)


def test_evaluate_population_records_every_member():
    algo = make_algo(budget=8)
    algo.generate_population(8)
    algo.evaluate_population(real_fitness, None)
    assert len(algo.population_metadata) == 8
    for meta in algo.population_metadata.values():
        assert set(meta) == {"id", "architecture", "fitness"}
        assert isinstance(meta["fitness"], EvaluationResult)


def test_evaluate_population_invokes_callback_per_member():
    algo = make_algo(budget=5)
    algo.generate_population(5)
    calls = []
    algo.evaluate_population(real_fitness, lambda n, a, r: calls.append(a))
    assert len(calls) == 5


def test_evaluate_empty_population():
    algo = make_algo()
    algo.generate_population(0)
    algo.evaluate_population(real_fitness, None)
    assert algo.population_metadata == {}


# ===========================================================================
# 3. SELECTION  (select_best_architecture / roulette / tournament)
# ===========================================================================
def test_select_best_architecture_picks_max_fitness():
    algo = make_algo()
    best = arch_with_layers(3)
    seed_population(algo, [(arch_with_layers(1), 0.2),
                           (best, 0.9),
                           (arch_with_layers(2), 0.5)])
    algo.select_best_architecture()
    assert algo.best_architecture is best


def test_select_best_architecture_empty_population():
    algo = make_algo()
    algo.population = []
    algo.population_metadata = {}
    algo.select_best_architecture()
    assert algo.best_architecture is None


def test_tournament_selection_count_and_uniqueness():
    random.seed(SEED)
    algo = make_algo()
    archs = [arch_with_layers(i + 1) for i in range(6)]
    seed_population(algo, [(a, 0.1 * (i + 1)) for i, a in enumerate(archs)])
    selected = algo.tournament_selection(4)
    assert len(selected) == 4
    assert len({id(s) for s in selected}) == 4          # no duplicates
    for s in selected:
        assert any(s is a for a in archs)               # only real members


def test_tournament_selection_all_members():
    random.seed(SEED)
    algo = make_algo()
    archs = [arch_with_layers(i + 1) for i in range(4)]
    seed_population(algo, [(a, 0.25 * (i + 1)) for i, a in enumerate(archs)])
    selected = algo.tournament_selection(4)
    assert {id(s) for s in selected} == {id(a) for a in archs}


def test_tournament_selection_more_than_population_raises():
    """Error handling: asking for more than exist drains candidates -> empty
    tournament -> max() on empty. We assert the failure mode explicitly."""
    random.seed(SEED)
    algo = make_algo()
    archs = [arch_with_layers(i + 1) for i in range(3)]
    seed_population(algo, [(a, 0.3 * (i + 1)) for i, a in enumerate(archs)])
    with raises((ValueError, IndexError)):
        algo.tournament_selection(5)


def test_roulette_selection_count_and_subset():
    random.seed(SEED)
    algo = make_algo()
    archs = [arch_with_layers(i + 1) for i in range(6)]
    seed_population(algo, [(a, 0.1 * (i + 1)) for i, a in enumerate(archs)])
    selected = algo.roulette_wheel_selection(4)
    assert len(selected) == 4
    assert len({id(s) for s in selected}) == 4
    for s in selected:
        assert any(s is a for a in archs)


def test_roulette_selection_biased_toward_high_fitness():
    """With one high-fitness and one zero-fitness member, single-draw roulette
    should overwhelmingly pick the high one."""
    random.seed(SEED)
    high_hits = 0
    trials = 200
    for _ in range(trials):
        algo = make_algo()
        high = arch_with_layers(3)
        low = arch_with_layers(1)
        seed_population(algo, [(high, 1.0), (low, 0.0)])
        picked = algo.roulette_wheel_selection(1)
        if picked and picked[0] is high:
            high_hits += 1
    assert high_hits > trials * 0.9


def test_roulette_all_zero_fitness_does_not_crash():
    random.seed(SEED)
    algo = make_algo()
    archs = [arch_with_layers(2) for _ in range(4)]
    seed_population(algo, [(a, 0.0) for a in archs])
    selected = algo.roulette_wheel_selection(2)
    assert len(selected) <= 2
    for s in selected:
        assert any(s is a for a in archs)


# ===========================================================================
# 4. MUTATION  (create_layer / global / inner-layer / mutate_architecture)
# ===========================================================================
def test_create_layer_is_valid_and_in_space():
    random.seed(SEED)
    algo = make_algo()
    space = make_space()
    for _ in range(30):
        layer = algo.create_layer()
        assert_layer_in_space(layer, space)


def test_create_layer_excludes_given_type():
    random.seed(SEED)
    algo = make_algo()
    for _ in range(30):
        layer = algo.create_layer(exclude_type="CNN")
        assert layer.layer_type != "CNN"


def test_global_mutation_does_not_mutate_parent():
    random.seed(SEED)
    algo = make_algo()
    parent = arch_with_layers(3)
    before = parent.to_json()
    child = algo.global_mutation(parent)
    assert parent.to_json() == before          # parent untouched (deepcopy)
    assert child is not parent


def test_global_mutation_keeps_num_layers_consistent():
    random.seed(SEED)
    algo = make_algo()
    for _ in range(50):
        parent = arch_with_layers(random.randint(1, 4))
        child = algo.global_mutation(parent)
        assert_structural(child)               # num_layers == len(layers)
        assert_buildable(child)


def test_inner_layer_mutation_produces_valid_buildable_arch():
    random.seed(SEED)
    algo = make_algo()
    for layer_type in ("CNN", "LSTM", "GRU"):
        parent = arch_with_layers(3, layer_type=layer_type)
        child = algo.inner_layer_mutation(parent)
        assert_structural(child)
        assert_buildable(child)
        assert parent.to_json() == arch_with_layers(3, layer_type=layer_type).to_json()


def test_inner_layer_mutation_on_empty_arch_raises():
    """Error handling: no layer to pick from -> randint(0, -1) must fail."""
    algo = make_algo()
    empty = arch_with_layers(0)
    with raises((ValueError, IndexError)):
        algo.inner_layer_mutation(empty)


def test_mutate_architecture_result_is_valid():
    random.seed(SEED)
    algo = make_algo()
    for _ in range(50):
        parent = arch_with_layers(random.randint(1, 5),
                                  layer_type=random.choice(["CNN", "LSTM", "GRU"]))
        child = algo.mutate_architecture(parent)
        assert_structural(child)
        assert_buildable(child)


def test_mutated_candidates_stay_in_search_space():
    """Stricter: mutated candidates should also respect declared bounds."""
    random.seed(SEED)
    algo = make_algo()
    for _ in range(50):
        parent = arch_with_layers(random.randint(1, 4))
        child = algo.mutate_architecture(parent)
        assert_in_search_space(child)


# ===========================================================================
# 5. CROSSOVER  (single-point / uniform / dispatch)
# ===========================================================================
def test_single_point_crossover_children_are_valid():
    random.seed(SEED)
    algo = make_algo()
    for n1, n2 in [(1, 1), (2, 3), (5, 5), (1, 5)]:
        p1 = arch_with_layers(n1, layer_type="CNN")
        p2 = arch_with_layers(n2, layer_type="GRU")
        c1, c2 = algo.single_point_crossover(p1, p2)
        for child in (c1, c2):
            assert_structural(child)
            assert_buildable(child)


def test_single_point_crossover_children_independent_of_parents():
    """Children must not alias parent layer objects, or later mutation of a
    child would corrupt a parent still in the population."""
    algo = make_algo()
    p1 = arch_with_layers(3, layer_type="CNN")
    p2 = arch_with_layers(3, layer_type="GRU")
    c1, _ = algo.single_point_crossover(p1, p2)
    before = p1.to_json()
    for layer in c1.layers:
        layer.hidden_size = 999
    assert p1.to_json() == before, "child shares layer objects with parent"


def test_single_point_crossover_with_empty_parent():
    algo = make_algo()
    p1 = arch_with_layers(0)
    p2 = arch_with_layers(3)
    c1, c2 = algo.single_point_crossover(p1, p2)
    assert_structural(c1)
    assert_structural(c2)


def test_uniform_crossover_returns_two_valid_children():
    random.seed(SEED)
    algo = make_algo()
    p1 = arch_with_layers(3, layer_type="CNN")
    p2 = arch_with_layers(4, layer_type="LSTM")
    result = algo.uniform_crossover(p1, p2)
    assert result is not None, "uniform_crossover is unimplemented"
    c1, c2 = result
    for child in (c1, c2):
        assert_structural(child)
        assert_buildable(child)


def test_crossover_dispatch_returns_two_valid_children():
    random.seed(SEED)
    algo = make_algo()
    p1 = arch_with_layers(3, layer_type="CNN")
    p2 = arch_with_layers(3, layer_type="GRU")
    result = algo.crossover(p1, p2)
    assert result is not None, "crossover() returns None"
    c1, c2 = result
    for child in (c1, c2):
        assert_structural(child)
        assert_buildable(child)


# ===========================================================================
# 6. INTEGRATION  (execute / search -- everything together)
# ===========================================================================
def test_execute_without_mutation_runs_clean():
    random.seed(SEED)
    algo = make_algo(budget=6)
    algo.execute(
        number_of_generations=3, num_selections=4,
        mutation_rate=0.0, crossover_rate=0.5,
        population_size=6, evaluate_fn=real_fitness, on_evaluation=None,
    )
    assert algo.best_architecture is not None
    assert_structural(algo.best_architecture)
    assert_buildable(algo.best_architecture)


def test_execute_with_mutation_tracks_mutated_children():
    """Forcing mutation exercises the path where children are NOT in
    population_metadata -- a lookup that must not KeyError."""
    random.seed(SEED)
    algo = make_algo(budget=6)
    algo.execute(
        number_of_generations=2, num_selections=4,
        mutation_rate=1.0, crossover_rate=0.5,
        population_size=6, evaluate_fn=real_fitness, on_evaluation=None,
    )


def test_search_end_to_end_budget_variations():
    for budget in (1, 4, 10):
        random.seed(SEED)
        algo = make_algo(budget=budget)
        algo.search(real_fitness, None)
        assert algo.best_architecture is not None
        assert_buildable(algo.best_architecture)


def test_full_ga_every_metadata_candidate_is_valid():
    """After a full run, every architecture the GA is holding must be valid."""
    random.seed(SEED)
    algo = make_algo(budget=10)
    algo.search(real_fitness, None)
    for meta in algo.population_metadata.values():
        assert_structural(meta["architecture"])
        assert_buildable(meta["architecture"])


# ===========================================================================
# 7. STATE / ISOLATION
# ===========================================================================
def test_class_attribute_state_is_not_shared():
    """population_metadata is declared at class scope; two raw instances must
    not share it (they currently do)."""
    a = Neuroevolution(make_space(), budget=2)
    b = Neuroevolution(make_space(), budget=2)
    a.population_metadata[id(a)] = {"marker": True}
    assert id(a) not in b.population_metadata, "instances share population_metadata"


# ---------------------------------------------------------------------------
# Plain-python runner (so the file works without pytest installed)
# ---------------------------------------------------------------------------
def _run_all():
    import traceback

    tests = sorted(
        (name, fn) for name, fn in globals().items()
        if name.startswith("test_") and callable(fn)
    )
    passed, failures = 0, []
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001 - test harness
            failures.append((name, traceback.format_exc()))
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")

    print(f"\n{passed} passed, {len(failures)} failed, {len(tests)} total")
    for name, tb in failures:
        print("=" * 72)
        print(name)
        print(tb)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_run_all())
