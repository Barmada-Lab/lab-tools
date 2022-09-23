import pathlib

from .experiment import Experiment
from .legacy import LegacyLoader

def load_experiment(path: pathlib.Path) -> Experiment | None:
    """
    auto-detects experiment structure and loads accordingly
    """
    loader = None
    if loader is None:
        loader = LegacyLoader(path)
        return loader.load()

    #default
    return None
