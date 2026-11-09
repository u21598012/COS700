
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
    return _ACTIVATIONS.get(name, nn.ReLU)()


_POOLS = {
    "max": nn.MaxPool1d,
    "avg": nn.AvgPool1d,
    "average": nn.AvgPool1d,
    "mean": nn.AvgPool1d,
}


def make_pool(name, kernel_size=2):
   
    if not name:
        return None
    cls = _POOLS.get(str(name).lower())
    if cls is None:
        return None
    return cls(kernel_size=kernel_size, stride=kernel_size)


def _apply_pool(pool, x):
   
    if pool is None:
        return x
    kernel = pool.kernel_size
    if isinstance(kernel, (tuple, list)):  
        kernel = kernel[0]
    if x.shape[-1] < kernel:
        return x
    return pool(x)


class CNNBlock(nn.Module):
 

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

    def forward(self, x):                     
        x = self.conv(x)                       
        x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = _apply_pool(self.pool, x)          
        return x


class _RecurrentBlock(nn.Module):


    rnn_cls = None  

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

    def forward(self, x):                     
        x = x.transpose(1, 2)                 
        x, _ = self.rnn(x)                    
        x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.transpose(1, 2)                  
        x = _apply_pool(self.pool, x)          
        return x


class LSTMBlock(_RecurrentBlock):
    rnn_cls = nn.LSTM


class GRUBlock(_RecurrentBlock):
    rnn_cls = nn.GRU


_BLOCK_REGISTRY = {
    "CNN": CNNBlock,
    "LSTM": LSTMBlock,
    "GRU": GRUBlock,
}


class ECGModel(nn.Module):

    def __init__(self, arch: ArchitectureConfig, input_dim: int, num_classes: int):
        super().__init__()
        blocks = []
        in_channels = 1                        
        for layer in arch.layers:
            block_cls = _BLOCK_REGISTRY.get(layer.layer_type)
            if block_cls is None:
                raise ValueError(
                    f"Unknown layer_type {layer.layer_type!r}; "
                    f"expected one of {sorted(_BLOCK_REGISTRY)}."
                )
            block = block_cls(layer, in_channels)
            blocks.append(block)
            in_channels = block.out_channels   
        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(in_channels, num_classes)

    def forward(self, x):                     
        x = x.unsqueeze(1)                    
        for block in self.blocks:
            x = block(x)                       
        x = self.pool(x).squeeze(-1)          
        return self.classifier(x)              


def build_model(arch: ArchitectureConfig, input_dim: int, num_classes: int) -> nn.Module:
  
    return ECGModel(arch, input_dim, num_classes)
