from pathlib import Path

from dask.distributed import Client, LocalCluster

from cytomancer.celery import app, CytomancerTask
from cytomancer.experiment import ExperimentType


@app.task(bind=True)
def run_pultra_survival(self: CytomancerTask, experiment_path: str, experiment_type: ExperimentType, svm_model_path: str, save_annotations: bool):
    from .pultra_survival import run
    client = Client(LocalCluster(n_workers=8, threads_per_worker=3))
    run(Path(experiment_path), experiment_type, Path(svm_model_path), save_annotations)
    client.shutdown()
    client.close()
