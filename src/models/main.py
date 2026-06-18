"""Permutation smoke test for the block builder.

Builds a model for EVERY ordered combination of block types and runs a forward
(and a quick backward) pass on a dummy batch, asserting the output is the
expected ``(batch, num_classes)`` logits. This proves any sequence the search
space can sample produces a valid, trainable model.

With DEPTHS = [1, 2, 3] and the 3 block types that is 3 + 3^2 + 3^3 = 39 models;
restrict to DEPTHS = [3] for the headline 3x3x3 = 27 combinations.

Run it (from the repo root, using the project venv):

    ./venv/Scripts/python.exe src/models/main.py
"""

import itertools
import sys
from pathlib import Path

import torch

# Make ``import src...`` work when run directly as a script.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.architecture import ArchitectureConfig, LayerConfig
from src.models.architecture_builder import build_model

# ---------------------------------------------------------------------------
# Adjust these and re-run.
# ---------------------------------------------------------------------------
BLOCK_TYPES = ["CNN", "LSTM", "GRU"]
DEPTHS = [1, 2, 3]          # sequence lengths to test; [3] -> the 3x3x3 = 27 case
POOLING_MODES = [None, "max", "avg"]   # every combo is tested under each
BATCH = 8
INPUT_DIM = 187             # MIT-BIH beat length
NUM_CLASSES = 5
HIDDEN_SIZE = 16
KERNEL_SIZE = 3
DROPOUT = 0.1
ACTIVATION = "relu"
BIDIRECTIONAL = True        # exercises the channel-doubling path for RNN blocks
# ---------------------------------------------------------------------------


def make_layer(layer_type, pooling):
    """One LayerConfig per type with every relevant field populated."""
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
    """Build, forward, and backward one architecture; return (ok, detail)."""
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

    # A quick backward proves the chain is differentiable end to end.
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
