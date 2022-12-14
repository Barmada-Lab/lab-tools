from typing import Callable

from improc.common.result import Result, Value
from improc.experiment.types import Dataset, Experiment, Image
from improc.processes.types import Task, TaskError


class Filter(Task):

    def __init__(self, predicate: Callable[[Image], bool]) -> None:
        super().__init__("filtered")
        self.predicate = predicate

    def process(self, dataset: Dataset, experiment: Experiment) -> Result[Dataset, TaskError]:
        new_dataset = experiment.new_dataset(self.output_label, overwrite=True)
        for image in dataset.images:
            if self.predicate(image):
                new_dataset.write_image(image.data, image.tags, image.axes)
        return Value(new_dataset)
