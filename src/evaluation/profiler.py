"""Resource monitoring (framework spec, Phase 11).

Provides point-in-time helpers (RAM / GPU) plus a background :class:`ResourceMonitor`
that samples CPU% and RAM while training runs and reports averages + peaks.
Sampling begins before training and ends after evaluation.
"""

import os
import threading

import psutil

PROCESS = psutil.Process(os.getpid())


def get_ram_usage():
    """Current process resident memory in MB."""
    return PROCESS.memory_info().rss / 1024 / 1024


# Backwards-compatible alias (older tests import this name).
get_memory_usage = get_ram_usage


def get_gpu_memory():
    """Peak GPU memory in MB for the current process, or 0.0 if no CUDA."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024 / 1024
    except Exception:
        pass
    return 0.0


def reset_gpu_peak():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


class ResourceMonitor:
    """Background sampler for CPU% and RAM (MB).

    Usage::

        monitor = ResourceMonitor(interval=1.0).start()
        ... train / evaluate ...
        stats = monitor.stop()   # dict of avg/peak cpu & ram, peak gpu
    """

    def __init__(self, interval=1.0):
        self.interval = interval
        self._thread = None
        self._stop_event = threading.Event()
        self._cpu_samples = []
        self._ram_samples = []

    def start(self):
        reset_gpu_peak()
        # Prime cpu_percent so the first real sample is meaningful.
        PROCESS.cpu_percent(interval=None)
        self._stop_event.clear()
        self._cpu_samples = []
        self._ram_samples = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        while not self._stop_event.is_set():
            try:
                self._cpu_samples.append(PROCESS.cpu_percent(interval=None))
                self._ram_samples.append(get_ram_usage())
            except Exception:
                pass
            self._stop_event.wait(self.interval)

    def stop(self):
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join(timeout=self.interval * 2 + 1)
            self._thread = None
        # Always capture at least one final sample.
        self._ram_samples.append(get_ram_usage())
        return self.summary()

    def summary(self):
        cpu = self._cpu_samples or [0.0]
        ram = self._ram_samples or [get_ram_usage()]
        return {
            "avg_cpu": sum(cpu) / len(cpu),
            "peak_cpu": max(cpu),
            "avg_ram_mb": sum(ram) / len(ram),
            "peak_ram_mb": max(ram),
            "peak_gpu_mb": get_gpu_memory(),
        }
