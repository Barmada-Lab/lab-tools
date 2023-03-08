import abc
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Hashable

from improc.common.result import Error, Result, Value, Ok

from improc.experiment import Experiment, Dataset
from improc.experiment.types import Image
from improc.utils import agg

from multiprocessing import Pool
from tqdm import tqdm

InputCollection = str
OutputCollection = str

class TaskError(abc.ABC):
    ...

class UnhandledShape(TaskError):
    ...

class Task(abc.ABC):

    def __init__(self, output_label: str) -> None:
        self.output_label = output_label

    @abc.abstractmethod
    def process(self, dataset: Dataset, experiment: Experiment) -> Result[Dataset, TaskError]:
        ...


class OneToOneTask(Task):

    def __init__(self, output_label: str, overwrite=False) -> None:
        super().__init__(output_label)
        self.overwrite=overwrite

    def filter(self, images: Iterable[Image]) -> Iterable[Image]:
        return images

    @abc.abstractmethod
    def transform(self, image: Image) -> Result[Image, TaskError]:
        ...

    def process(self, dataset: Dataset, experiment: Experiment) -> Result[Dataset, TaskError]:
        output_dataset = experiment.new_dataset(self.output_label, overwrite=self.overwrite)
        with Pool(4) as p:
            for result in tqdm(p.imap(self.transform, self.filter(dataset.images)), total=len(dataset.images), desc=self.__class__.__name__):
                match result:
                    case Value(image):
                        output_dataset.write_image2(image)
                    case Error(taskerr):
                        return Error(taskerr)
        return Value(output_dataset)

class ManyToOneTask(Task):

    def __init__(self, output_label: str, overwrite=False) -> None:
        super().__init__(output_label)
        self.overwrite = overwrite

    @abc.abstractmethod
    def group_pred(self, image: Image) -> Hashable:
        ...

    @abc.abstractmethod
    def transform(self, images: list[Image]) -> Result[Image, TaskError]:
        ...

    def process(self, dataset: Dataset, experiment: Experiment) -> Result[Dataset, TaskError]:
        output_dataset = experiment.new_dataset(self.output_label, overwrite=self.overwrite)
        groups = list(agg.groupby(dataset.images, self.group_pred).values())
        with Pool(4) as p:
            for result in tqdm(p.imap(self.transform, groups), total=len(groups), desc=self.__class__.__name__):
                match result:
                    case Value(image):
                        output_dataset.write_image2(image)
                    case Error(taskerr):
                        return Error(taskerr)
        return Value(output_dataset)


@dataclass
class InvalidInputLabel(TaskError):
    label: str

class Pipeline:
    def __init__(self, *args: Task) -> None:
        self.tasks = args

    def run(self, experiment: Experiment, input_label: str) -> Result[Ok, TaskError]:
        dataset = experiment.datasets[input_label]
        if dataset is None:
            return Error(InvalidInputLabel(input_label))
        for task in self.tasks:
            result = task.process(dataset, experiment)
            match result:
                case Value(result_dataset):
                    dataset = result_dataset
                case TaskError():
                    return result
        return Value(Ok())
