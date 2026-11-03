

import json
from dataclasses import asdict, dataclass, field
from typing import List, Optional


@dataclass
class LayerConfig:
    """Description of a single layer in the network."""

    layer_type: str               
    hidden_size: int
    dropout: float = 0.0
    activation: Optional[str] = None      
    kernel_size: Optional[int] = None     # used by CNN layers
    bidirectional: Optional[bool] = None  # used by recurrent layers
    pooling: Optional[str] = None         


@dataclass
class ArchitectureConfig:

    learning_rate: float
    num_layers: int
    layers: List[LayerConfig] = field(default_factory=list)

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
