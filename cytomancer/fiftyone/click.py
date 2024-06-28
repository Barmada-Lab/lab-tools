from pathlib import Path
import shutil

from dask.distributed import Client
import fiftyone as fo
import click

from cytomancer.click_utils import experiment_dir_argument


@click.command("launch-app")
def launch_app() -> None:
    fo.launch_app(address="0.0.0.0")  # type: ignore


@click.command("delete-dataset")
@click.argument("dataset_name")
def delete_dataset(dataset_name: str) -> None:
    dataset = fo.load_dataset(dataset_name)
    shutil.rmtree(dataset.info["media_dir"])  # type: ignore
    fo.delete_dataset(dataset_name)


@click.command("ingest", help="Ingest a CQ1 dataset into FiftyOne. Other formats are not yet supported.")
@experiment_dir_argument()
def ingest(experiment_dir: Path) -> None:
    with Client(n_workers=8, threads_per_worker=2) as client:  # noqa: F841
        from .ingest import ingest_cq1
        ingest_cq1(experiment_dir)


def register(cli: click.Group) -> None:
    @cli.group("fiftyone")
    def fiftyone() -> None:
        pass

    fiftyone.add_command(ingest)
    fiftyone.add_command(launch_app)
    fiftyone.add_command(delete_dataset)
