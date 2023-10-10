from typing import Callable

from improc.experiment.types import Dataset, Experiment, Image
from improc.processes.types import Task


class Filter(Task):

    def __init__(self, predicate: Callable[[Image], bool], overwrite=False) -> None:
        super().__init__("filtered")
        self.predicate = predicate
        self.overwrite = overwrite

    def process(self, dataset: Dataset, experiment: Experiment) -> Dataset:
        new_dataset = experiment.new_dataset(self.output_label, overwrite=self.overwrite)
        for image in dataset.images:
            if self.predicate(image):
                new_dataset.write_image(image.data, image.tags, image.axes)
        return new_dataset
