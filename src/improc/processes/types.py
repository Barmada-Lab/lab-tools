import abc
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Hashable


from improc.experiment.types import Image, Experiment, Dataset
from improc import agg

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
    def process(self, dataset: Dataset, experiment: Experiment) -> Dataset:
        ...


class OneToOneTask(Task):

    def __init__(self, output_label: str, overwrite=False) -> None:
        super().__init__(output_label)
        self.overwrite=overwrite

    def filter(self, images: Iterable[Image]) -> Iterable[Image]:
        return images

    @abc.abstractmethod
    def transform(self, image: Image) -> Image:
        ...

    def process(self, dataset: Dataset, experiment: Experiment) -> Dataset:
        output_dataset = experiment.new_dataset(self.output_label, overwrite=self.overwrite)
        with Pool(4) as p:
            for result in tqdm(p.imap(self.transform, self.filter(dataset.images)), total=len(dataset.images), desc=self.__class__.__name__):
                output_dataset.write_image2(result)
        return output_dataset

class ManyToOneTask(Task):

    def __init__(self, output_label: str, parallelism: int=1, overwrite=False) -> None:
        super().__init__(output_label)
        self.overwrite = overwrite
        self.parallelism = parallelism

    @abc.abstractmethod
    def group_pred(self, image: Image) -> Hashable:
        ...

    @abc.abstractmethod
    def transform(self, images: list[Image]) -> Image:
        ...

    def process(self, dataset: Dataset, experiment: Experiment) -> Dataset:
        output_dataset = experiment.new_dataset(self.output_label, overwrite=self.overwrite)
        groups = list(agg.groupby(dataset.images, self.group_pred).values())
        with Pool(self.parallelism) as p:
            for result in tqdm(p.imap(self.transform, groups), total=len(groups), desc=self.__class__.__name__):
                output_dataset.write_image2(result)
        return output_dataset


@dataclass
class InvalidInputLabel(TaskError):
    label: str

class Pipeline:
    def __init__(self, *args: Task) -> None:
        self.tasks = args

    def run(self, experiment: Experiment, input_label: str):
        dataset = experiment.datasets[input_label]
        if dataset is None:
            raise Exception(f"Invalid label {input_label}")
        for task in self.tasks:
            result = task.process(dataset, experiment)
            dataset = result
