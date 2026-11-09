

from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from src.evaluation.metrics import compute_metrics
from src.evaluation.profiler import ResourceMonitor
from src.evaluation.timing import Timer
from src.models.architecture_builder import build_model


@dataclass
class EvaluationResult:

    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    macro_f1: float = 0.0

    training_time: float = 0.0
    inference_time: float = 0.0

    avg_cpu: float = 0.0
    peak_cpu: float = 0.0
    avg_ram_mb: float = 0.0
    peak_ram_mb: float = 0.0
    peak_gpu_mb: float = 0.0

    peak_memory_mb: float = 0.0

    failed: bool = False
    error: Optional[str] = None

    predictions: Optional[np.ndarray] = field(default=None, repr=False)
    targets: Optional[np.ndarray] = field(default=None, repr=False)

    def to_record(self):
        d = asdict(self)
        d.pop("predictions", None)
        d.pop("targets", None)
        return d


def _train_one_epoch(model, loader, optimizer, criterion, device, max_batches=None):
    model.train()
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def _evaluate(model, loader, device, max_batches=None):
    model.eval()
    preds, targets = [], []
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device)
        logits = model(x)
        preds.append(logits.argmax(dim=1).cpu().numpy())
        targets.append(y.numpy())
    if not preds:
        return np.array([]), np.array([])
    return np.concatenate(preds), np.concatenate(targets)


def evaluate_architecture(architecture, train_loader, val_loader, config,
                          input_dim, num_classes, device="cpu"):

    training_cfg = config.get("training", {})
    epochs = int(training_cfg.get("epochs", 10))
    max_train_batches = training_cfg.get("max_train_batches")
    max_eval_batches = training_cfg.get("max_eval_batches")

    device = torch.device(device)
    monitor = ResourceMonitor(
        interval=config.get("resource_sample_interval", 0.5)
    ).start()
    timer = Timer()

    result = EvaluationResult()
    try:
        model = build_model(architecture, input_dim, num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=architecture.learning_rate)
        criterion = nn.CrossEntropyLoss()

        timer.begin()
        for _ in range(epochs):
            _train_one_epoch(model, train_loader, optimizer, criterion, device, max_train_batches)
        result.training_time = timer.stop()

        timer.begin()
        preds, targets = _evaluate(model, val_loader, device, max_eval_batches)
        result.inference_time = timer.stop()

        if len(preds):
            metrics = compute_metrics(targets, preds)
            result.accuracy = metrics["accuracy"]
            result.precision = metrics["precision"]
            result.recall = metrics["recall"]
            result.macro_f1 = metrics["macro_f1"]
            result.predictions = preds
            result.targets = targets
    except Exception as exc:  
        result.failed = True
        result.error = f"{type(exc).__name__}: {exc}"

    stats = monitor.stop()
    result.avg_cpu = stats["avg_cpu"]
    result.peak_cpu = stats["peak_cpu"]
    result.avg_ram_mb = stats["avg_ram_mb"]
    result.peak_ram_mb = stats["peak_ram_mb"]
    result.peak_gpu_mb = stats["peak_gpu_mb"]
    result.peak_memory_mb = max(stats["peak_ram_mb"], stats["peak_gpu_mb"])
    return result
