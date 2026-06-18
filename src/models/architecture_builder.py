"""Model builder: ArchitectureConfig -> torch.nn.Module (framework spec, Phase 9).

Search algorithms NEVER create PyTorch layers directly. They produce an
:class:`~src.models.architecture.ArchitectureConfig`; this module turns it into
an executable model.

----------------------------------------------------------------------------
The universal block interface ("the contract")
----------------------------------------------------------------------------
Every block (CNN / LSTM / GRU) speaks ONE common tensor format so they can be
stacked in *any* order without the caller knowing what each block is:

        UNIVERSAL FORMAT:  (batch, channels, length)

  * ``channels`` is the per-timestep feature dimension.
  * ``length``   is the sequence/time axis (the 187 ECG samples to start with).

A block receives a tensor in this format, internally transforms it into
whatever layout its underlying torch layer needs, processes it, then transforms
the result back into the universal format and returns it. A block changes
``channels`` (to ``out_channels``) and may shrink ``length`` (optional pooling),
but never breaks the format - so an arbitrary permutation of blocks always
lines up at runtime. ``length`` only ever decreases, and the final global pool
collapses whatever remains, so the classifier never depends on it.

  block.forward:      (batch, C_in, length)  ->  (batch, C_out, length)
  block.out_channels: the C_out the next block should expect.

The whole network is therefore:

  (batch, input_dim)
      -> reshape to (batch, 1, input_dim)            # 1 channel, length=input_dim
      -> block_1 -> block_2 -> ... -> block_n        # universal format throughout
      -> global average pool over length             # (batch, C_final)
      -> Linear classifier                           # (batch, num_classes)

Contract you must keep if you extend this:
  * input  to the built model:  (batch, input_dim).
  * output from it:             logits of shape (batch, num_classes).
"""

import torch.nn as nn

from src.models.architecture import ArchitectureConfig, LayerConfig


_ACTIVATIONS = {
    None: nn.ReLU,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "leaky_relu": nn.LeakyReLU,
}


def get_activation(name):
    """Map an activation name to a fresh nn.Module (defaults to ReLU)."""
    return _ACTIVATIONS.get(name, nn.ReLU)()


_POOLS = {
    "max": nn.MaxPool1d,
    "avg": nn.AvgPool1d,
    "average": nn.AvgPool1d,
    "mean": nn.AvgPool1d,
}


def make_pool(name, kernel_size=2):
    """Build an optional length-pooling module.

    Returns ``None`` when ``name`` is falsy/unknown (no pooling). Operates on the
    universal ``(B, C, L)`` layout and halves ``L`` by default (kernel/stride 2).
    """
    if not name:
        return None
    cls = _POOLS.get(str(name).lower())
    if cls is None:
        return None
    return cls(kernel_size=kernel_size, stride=kernel_size)


def _apply_pool(pool, x):
    """Apply a pool module on ``(B, C, L)``, skipping it if ``L`` is too short.

    Repeated pooling shrinks ``L``; guarding here means a deep stack can never
    fail by trying to pool a length-1 sequence.
    """
    if pool is None:
        return x
    kernel = pool.kernel_size
    if isinstance(kernel, (tuple, list)):   # AvgPool1d stores it as a tuple
        kernel = kernel[0]
    if x.shape[-1] < kernel:
        return x
    return pool(x)


# ===========================================================================
#  Blocks - each consumes and returns the universal (batch, channels, length)
# ===========================================================================
class CNNBlock(nn.Module):
    """1-D convolution block.

    Universal in/out: ``(batch, C, L)`` -> ``(batch, hidden_size, L)``.
    Conv1d already works in channel-first layout, so no transpose is needed.
    ``padding = kernel_size // 2`` keeps ``L`` unchanged for the odd kernel
    sizes used here (3 / 5 / 7).
    """

    def __init__(self, layer: LayerConfig, in_channels: int):
        super().__init__()
        self.out_channels = max(1, int(layer.hidden_size))
        kernel_size = int(layer.kernel_size) if layer.kernel_size else 3
        self.conv = nn.Conv1d(
            in_channels,
            self.out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.activation = get_activation(layer.activation)
        self.dropout = nn.Dropout(float(layer.dropout)) if layer.dropout else None
        self.pool = make_pool(layer.pooling)

    def forward(self, x):                      # x: (B, C_in, L)
        x = self.conv(x)                       # -> (B, C_out, L)
        x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = _apply_pool(self.pool, x)          # optional -> (B, C_out, L // 2)
        return x


class _RecurrentBlock(nn.Module):
    """Shared LSTM/GRU logic; subclasses set ``rnn_cls``.

    The universal format is channel-first ``(B, C, L)`` but torch RNNs want
    ``(B, L, features)`` with ``batch_first=True``. So we transpose in, run the
    RNN over the length axis (each of the ``L`` positions is one timestep), then
    transpose the output back to channel-first. ``out_channels`` doubles when
    bidirectional because the two directions are concatenated.
    """

    rnn_cls = None  # set by subclass

    def __init__(self, layer: LayerConfig, in_channels: int):
        super().__init__()
        hidden_size = max(1, int(layer.hidden_size))
        bidirectional = bool(layer.bidirectional)
        self.out_channels = hidden_size * (2 if bidirectional else 1)
        self.rnn = self.rnn_cls(
            input_size=in_channels,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.activation = get_activation(layer.activation)
        self.dropout = nn.Dropout(float(layer.dropout)) if layer.dropout else None
        self.pool = make_pool(layer.pooling)

    def forward(self, x):                      # x: (B, C_in, L)
        x = x.transpose(1, 2)                  # -> (B, L, C_in)  (features last)
        x, _ = self.rnn(x)                     # -> (B, L, H * dirs)
        x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.transpose(1, 2)                  # -> (B, H * dirs, L)  back to universal
        x = _apply_pool(self.pool, x)          # optional -> (B, H * dirs, L // 2)
        return x


class LSTMBlock(_RecurrentBlock):
    rnn_cls = nn.LSTM


class GRUBlock(_RecurrentBlock):
    rnn_cls = nn.GRU


# Registry: layer_type string -> block class. Extend here to add new block kinds.
_BLOCK_REGISTRY = {
    "CNN": CNNBlock,
    "LSTM": LSTMBlock,
    "GRU": GRUBlock,
}


# ===========================================================================
#  The assembled network
# ===========================================================================
class ECGModel(nn.Module):
    """Builds the full block chain from an ArchitectureConfig.

    Input  : ``(batch, input_dim)``  (a raw ECG beat, e.g. 187 samples).
    Output : ``(batch, num_classes)`` logits.
    """

    def __init__(self, arch: ArchitectureConfig, input_dim: int, num_classes: int):
        super().__init__()
        blocks = []
        in_channels = 1                        # the beat starts as a single channel
        for layer in arch.layers:
            block_cls = _BLOCK_REGISTRY.get(layer.layer_type)
            if block_cls is None:
                raise ValueError(
                    f"Unknown layer_type {layer.layer_type!r}; "
                    f"expected one of {sorted(_BLOCK_REGISTRY)}."
                )
            block = block_cls(layer, in_channels)
            blocks.append(block)
            in_channels = block.out_channels   # feed C_out into the next block
        self.blocks = nn.ModuleList(blocks)
        # Global average pool over the length axis collapses (B, C, L) -> (B, C),
        # keeping the classifier independent of the (preserved) sequence length.
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(in_channels, num_classes)

    def forward(self, x):                      # x: (B, input_dim)
        x = x.unsqueeze(1)                     # -> (B, 1, input_dim)  universal format
        for block in self.blocks:
            x = block(x)                       # (B, C, L) -> (B, C', L)
        x = self.pool(x).squeeze(-1)           # (B, C_final, L) -> (B, C_final)
        return self.classifier(x)              # (B, num_classes)


def build_model(arch: ArchitectureConfig, input_dim: int, num_classes: int) -> nn.Module:
    """Convert an architecture description into a trainable model.

    Dispatches each :class:`LayerConfig` to its block in ``_BLOCK_REGISTRY`` and
    chains them through the universal ``(batch, channels, length)`` format.
    """
    return ECGModel(arch, input_dim, num_classes)
