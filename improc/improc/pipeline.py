from .task import Task
from .utils.arr_experiment import Experiment, Dataset

class Pipeline:

    def __init__(self, *tasks: Task):
        self.tasks = tasks

    def dryrun(self):
        pass

    def run(self, experiment: Experiment, dataset: Dataset):
        for task in self.tasks:
            task.run()
