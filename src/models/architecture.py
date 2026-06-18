"""Canonical architecture representation shared by every search algorithm.

Random Search, Bayesian Optimization, Neuroevolution and the Hybrid must all
produce/consume this exact schema (framework spec, Phase 8). This keeps the
model builder and evaluator algorithm-agnostic.
"""

import json
from dataclasses import asdict, dataclass, field
from typing import List, Optional


@dataclass
class LayerConfig:
    """Description of a single layer in the network."""

    layer_type: str               # e.g. "CNN", "LSTM", "GRU"
    hidden_size: int
    dropout: float = 0.0
    activation: Optional[str] = None      # e.g. "relu", "tanh"
    kernel_size: Optional[int] = None     # used by CNN layers
    bidirectional: Optional[bool] = None  # used by recurrent layers
    pooling: Optional[str] = None         # None | "max" | "avg" - downsample length


@dataclass
class ArchitectureConfig:
    """A full candidate architecture."""

    learning_rate: float
    num_layers: int
    layers: List[LayerConfig] = field(default_factory=list)

    # ---- serialisation helpers -------------------------------------------
    def to_dict(self):
        return asdict(self)

    def to_json(self, indent=None):
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data):
        layers = [LayerConfig(**layer) for layer in data.get("layers", [])]
        return cls(
            learning_rate=data["learning_rate"],
            num_layers=data["num_layers"],
            layers=layers,
        )

    @classmethod
    def from_json(cls, text):
        return cls.from_dict(json.loads(text))
