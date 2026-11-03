

import itertools
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.architecture import ArchitectureConfig, LayerConfig
from src.models.architecture_builder import build_model

BLOCK_TYPES = ["CNN", "LSTM", "GRU"]
DEPTHS = [1, 2, 3]          
POOLING_MODES = [None, "max", "avg"]  
BATCH = 8
INPUT_DIM = 187            
NUM_CLASSES = 5
HIDDEN_SIZE = 16
KERNEL_SIZE = 3
DROPOUT = 0.1
ACTIVATION = "relu"
BIDIRECTIONAL = True        

def make_layer(layer_type, pooling):
    return LayerConfig(
        layer_type=layer_type,
        hidden_size=HIDDEN_SIZE,
        dropout=DROPOUT,
        activation=ACTIVATION,
        kernel_size=KERNEL_SIZE,
        bidirectional=BIDIRECTIONAL,
        pooling=pooling,
    )


def check_combo(types, pooling):
    arch = ArchitectureConfig(
        learning_rate=1e-3,
        num_layers=len(types),
        layers=[make_layer(t, pooling) for t in types],
    )
    model = build_model(arch, INPUT_DIM, NUM_CLASSES)

    x = torch.randn(BATCH, INPUT_DIM)
    logits = model(x)

    expected = (BATCH, NUM_CLASSES)
    if tuple(logits.shape) != expected:
        return False, f"bad output shape {tuple(logits.shape)} (expected {expected})"
    if not torch.isfinite(logits).all():
        return False, "non-finite values in output"

    loss = logits.sum()
    loss.backward()

    n_params = sum(p.numel() for p in model.parameters())
    return True, f"out {tuple(logits.shape)}, params {n_params:,}"


def main():
    combos = []
    for depth in DEPTHS:
        combos.extend(itertools.product(BLOCK_TYPES, repeat=depth))

    total = len(combos) * len(POOLING_MODES)
    print(f"Testing {total} models: {len(combos)} type combinations "
          f"(depths={DEPTHS}, types={BLOCK_TYPES}) x pooling {POOLING_MODES}\n")

    passed, failed = 0, 0
    for pooling in POOLING_MODES:
        print(f"--- pooling = {pooling!r} ---")
        for types in combos:
            ok, detail = check_combo(types, pooling)
            tag = "OK  " if ok else "FAIL"
            print(f"  [{tag}] {' -> '.join(types):28s} | {detail}")
            if ok:
                passed += 1
            else:
                failed += 1
        print()

    print(f"{passed}/{total} passed, {failed} failed.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
