"""Configuration loading.

The whole framework reads its settings from the four YAML files in ``configs/``.
Nothing else in the codebase should call ``yaml`` directly - use these helpers so
config resolution stays in one place.
"""

import copy

import yaml

from src.utils import paths


def load_yaml(path):
    """Load a single YAML file into a dict."""
    with open(paths.resolve(path), "r") as f:
        return yaml.safe_load(f)


def load_configs(config_dir=None):
    """Load every framework config into one dict.

    Returns a dict with keys: ``experiment``, ``search_space``, ``dataset``,
    ``tracking``.
    """
    config_dir = paths.CONFIG_DIR if config_dir is None else paths.resolve(config_dir)
    return {
        "experiment": load_yaml(config_dir / "experiment.yaml"),
        "search_space": load_yaml(config_dir / "search_space.yaml"),
        "dataset": load_yaml(config_dir / "dataset.yaml"),
        "tracking": load_yaml(config_dir / "tracking.yaml"),
    }


def apply_overrides(config, overrides):
    """Return a deep-copied ``config`` with ``overrides`` merged in recursively.

    Useful for notebook "smoke runs" where you want to shrink the budget/epochs
    without editing the YAML files, e.g.::

        cfg = apply_overrides(cfg, {"experiment": {"budget": {"evaluations": 3}}})
    """
    merged = copy.deepcopy(config)

    def _merge(dst, src):
        for key, value in src.items():
            if isinstance(value, dict) and isinstance(dst.get(key), dict):
                _merge(dst[key], value)
            else:
                dst[key] = value

    _merge(merged, overrides)
    return merged
