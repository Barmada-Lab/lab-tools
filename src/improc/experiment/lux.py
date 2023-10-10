import pathlib

from .types import Experiment, DirDataset, TimeseriesDirDataset

class LuxLoader:

    def __init__(self, experiment_path: pathlib.Path, scratch_path: pathlib.Path) -> None:
        super().__init__()
        self.experiment_path = experiment_path
        self.scratch_path = scratch_path

    def load(self) -> Experiment:
        raw_dataset = TimeseriesDirDataset("raw_imgs", self.experiment_path / "raw_imgs")
        datasets = {"raw_imgs": raw_dataset}
        for path in self.scratch_path.glob("*"):
            if path.is_dir():
                label = path.name
                datasets[label] = DirDataset(label, path)

        return Experiment(
            label = self.experiment_path.name,
            datasets = datasets,
            experiment_dir = self.experiment_path,
            scratch_dir = self.scratch_path,
        )
