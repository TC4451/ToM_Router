"""Config loading utilities."""

from pathlib import Path
from omegaconf import OmegaConf


def load_config(path: str) -> OmegaConf:
    """Load a YAML config file."""
    return OmegaConf.load(path)


def merge_configs(*configs) -> OmegaConf:
    """Merge multiple configs (later ones override earlier)."""
    return OmegaConf.merge(*configs)
