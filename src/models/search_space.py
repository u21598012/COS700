

import random

from src.models.architecture import ArchitectureConfig, LayerConfig
from src.utils.config import load_yaml
from src.utils import paths


class SearchSpace:

    def __init__(self, space):
        self.space = space
        self._validate()

    @classmethod
    def from_yaml(cls, path=None):
        path = path or (paths.CONFIG_DIR / "search_space.yaml")
        return cls(load_yaml(path))

    def _validate(self):
        g = self.space.get("global", {})
        layer = self.space.get("layer", {})

        _check_range(g.get("learning_rate"), "global.learning_rate")
        _check_range(g.get("num_layers"), "global.num_layers")
        _check_range(layer.get("hidden_size"), "layer.hidden_size")
        _check_range(layer.get("dropout"), "layer.dropout")

        if not layer.get("type", {}).get("choices"):
            raise ValueError("search_space.layer.type.choices must be a non-empty list")

    def sample_learning_rate(self):
        r = self.space["global"]["learning_rate"]
        return random.uniform(r["min"], r["max"])

    def sample_num_layers(self):
        r = self.space["global"]["num_layers"]
        return random.randint(r["min"], r["max"])

    def sample_layer_type(self):
        return random.choice(self.space["layer"]["type"]["choices"])

    def sample_hidden_size(self):
        r = self.space["layer"]["hidden_size"]
        return random.randint(r["min"], r["max"])

    def sample_dropout(self):
        r = self.space["layer"]["dropout"]
        return round(random.uniform(r["min"], r["max"]), 4)

    def sample_activation(self):
        choices = self.space.get("layer", {}).get("activation", {}).get("choices")
        return random.choice(choices) if choices else None

    def sample_kernel_size(self):
        choices = self.space.get("layer", {}).get("kernel_size", {}).get("choices")
        return random.choice(choices) if choices else None

    def sample_bidirectional(self):
        choices = self.space.get("layer", {}).get("bidirectional", {}).get("choices")
        return random.choice(choices) if choices is not None else None

    def sample_pooling(self):
        choices = self.space.get("layer", {}).get("pooling", {}).get("choices")
        return random.choice(choices) if choices else None

    def sample_layer(self):
        return LayerConfig(
            layer_type=self.sample_layer_type(),
            hidden_size=self.sample_hidden_size(),
            dropout=self.sample_dropout(),
            activation=self.sample_activation(),
            kernel_size=self.sample_kernel_size(),
            bidirectional=self.sample_bidirectional(),
            pooling=self.sample_pooling(),
        )

    def sample_architecture(self):
        """Draw one complete random :class:`ArchitectureConfig`."""
        num_layers = self.sample_num_layers()
        return ArchitectureConfig(
            learning_rate=self.sample_learning_rate(),
            num_layers=num_layers,
            layers=[self.sample_layer() for _ in range(num_layers)],
        )


def _check_range(spec, name):
    if not spec or "min" not in spec or "max" not in spec:
        raise ValueError(f"search_space.{name} must define 'min' and 'max'")
    if spec["min"] > spec["max"]:
        raise ValueError(f"search_space.{name}: min ({spec['min']}) > max ({spec['max']})")
