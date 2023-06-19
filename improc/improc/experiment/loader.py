import pathlib

from .types import Experiment
from .legacy.loader import LegacyLoader
from .lux import LuxLoader

def load_experiment(path: pathlib.Path, scratch_dir: pathlib.Path) -> Experiment:
    try:
        loader = LegacyLoader(path, scratch_dir)
    except:
        loader = LuxLoader(path, scratch_dir)

    return loader.load()