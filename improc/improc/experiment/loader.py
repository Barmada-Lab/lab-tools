import pathlib

from .types import Experiment
from .legacy.loader import LegacyLoader

def load_experiment(path: pathlib.Path, scratch_dir: pathlib.Path) -> Experiment:
    """
    auto-detects experiment structure and loads accordingly
    """
    loader = LegacyLoader(path, scratch_dir)
    return loader.load()
