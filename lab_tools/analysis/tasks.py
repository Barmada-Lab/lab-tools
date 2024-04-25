from pathlib import Path

from dask.distributed import Client, LocalCluster

from lab_tools.celery import app, JobQueueTask
from lab_tools.experiment import ExperimentType


@app.task(bind=True)
def run_pultra_survival(self: JobQueueTask, experiment_path: str, experiment_type: ExperimentType, save_annotations: bool):
    from .pultra_survival import run
    client = Client(LocalCluster(n_workers=8, threads_per_worker=2))
    run(Path(experiment_path), experiment_type, save_annotations)
    client.shutdown()
    client.close()
