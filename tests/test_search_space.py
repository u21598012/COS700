"""Search space sampling tests."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.architecture import ArchitectureConfig, LayerConfig
from src.models.search_space import SearchSpace

SPACE = {
    "global": {
        "learning_rate": {"min": 0.0001, "max": 0.01},
        "num_layers": {"min": 1, "max": 4},
    },
    "layer": {
        "type": {"choices": ["CNN", "LSTM", "GRU"]},
        "hidden_size": {"min": 16, "max": 64},
        "dropout": {"min": 0.0, "max": 0.5},
        "activation": {"choices": ["relu", "tanh"]},
        "kernel_size": {"choices": [3, 5]},
        "bidirectional": {"choices": [True, False]},
    },
}


def test_sample_architecture_is_valid():
    space = SearchSpace(SPACE)
    for _ in range(50):
        arch = space.sample_architecture()
        assert isinstance(arch, ArchitectureConfig)
        assert 1 <= arch.num_layers <= 4
        assert len(arch.layers) == arch.num_layers
        assert 0.0001 <= arch.learning_rate <= 0.01
        for layer in arch.layers:
            assert isinstance(layer, LayerConfig)
            assert layer.layer_type in {"CNN", "LSTM", "GRU"}
            assert 16 <= layer.hidden_size <= 64
            assert 0.0 <= layer.dropout <= 0.5


def test_roundtrip_serialisation():
    space = SearchSpace(SPACE)
    arch = space.sample_architecture()
    restored = ArchitectureConfig.from_json(arch.to_json())
    assert restored.to_dict() == arch.to_dict()


def test_invalid_range_rejected():
    bad = {k: dict(v) for k, v in SPACE.items()}
    bad["global"] = {"learning_rate": {"min": 1, "max": 0}, "num_layers": {"min": 1, "max": 2}}
    try:
        SearchSpace(bad)
    except ValueError:
        return
    raise AssertionError("expected ValueError for min > max")


if __name__ == "__main__":
    test_sample_architecture_is_valid()
    test_roundtrip_serialisation()
    test_invalid_range_rejected()
    print("test_search_space: OK")
