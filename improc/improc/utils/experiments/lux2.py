import pathlib

from common.experiments.experiment import Experiment, ExperimentLoader


class Lux2Loader(ExperimentLoader):

    def __init__(self, path: pathlib.Path) -> None:
        super().__init__()
        self.path = path

    def load(self) -> Experiment:
        return Experiment("",[])
