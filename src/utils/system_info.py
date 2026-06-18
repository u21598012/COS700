"""Collect host/runtime metadata so every experiment is reproducible.

The returned dict is stored as JSON on the ``experiments`` table and can be
logged to MLflow as tags/params.
"""

import platform
import sys

import psutil


def get_system_info():
    """Return a JSON-serialisable dict describing the current machine."""
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": sys.version.split()[0],
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "total_ram_mb": round(psutil.virtual_memory().total / 1024 / 1024, 2),
    }

    # Torch / CUDA details are optional - never let a missing import break setup.
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_name"] = torch.cuda.get_device_name(0)
        else:
            info["gpu_count"] = 0
            info["gpu_name"] = None
    except Exception as exc:  # pragma: no cover - defensive
        info["torch_version"] = None
        info["cuda_available"] = False
        info["gpu_count"] = 0
        info["gpu_name"] = None
        info["torch_import_error"] = str(exc)

    return info


def get_device(prefer="auto", warn=True):
    """Return the torch device string to train on. Prefers GPU by default.

    ``prefer`` may be ``"auto"`` (GPU when present), ``"cpu"`` or ``"cuda"``.
    When GPU is wanted but unavailable, prints a loud warning (instead of
    silently using the CPU) explaining the likely cause.
    """
    import torch

    if prefer == "cpu":
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"

    if warn:
        lines = [
            "=" * 70,
            "[device] GPU requested but CUDA is NOT available -> falling back to CPU.",
        ]
        if torch.version.cuda is None:
            lines += [
                f"  Installed torch is the CPU-only build ('{torch.__version__}').",
                "  You have a CUDA GPU, so to actually use it reinstall the CUDA build:",
                "    pip uninstall -y torch torchvision torchaudio",
                "    pip install torch torchvision torchaudio \\",
                "      --index-url https://download.pytorch.org/whl/cu128",
                "  (RTX 50-series / Blackwell needs cu128+; try cu129 if cu128 fails.)",
            ]
        else:
            lines.append("  torch has CUDA support but no GPU/driver was detected.")
        lines.append("=" * 70)
        print("\n".join(lines))

    return "cpu"
