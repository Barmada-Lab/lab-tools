from pathlib import Path

import click

from improc.experiment import Experiment, loader
from improc.processes import Pipeline, Stitch, Stack

def prep_masa(experiment: Experiment, collection: str = "raw_imgs"):
    pipeline = Pipeline(
        Stitch(),
        Stack(register=False)
    )
    pipeline.run(experiment, collection)

@click.command("masa-prep")
@click.argument('experiment_dir', type=click.Path(exists=True, path_type=Path))
@click.option('--scratch-dir', type=click.Path(path_type=Path), default=None)
@click.option('--collection', type=str, default="raw_imgs")
@click.option('--preprocess', is_flag=True, default=False)
def cli_entry(experiment_dir: Path, scratch_dir: Path, collection: str):
    scratch_dir = scratch_dir if scratch_dir is not None else experiment_dir / "processed_imgs"
    experiment = loader.load_experiment(experiment_dir, scratch_dir)
    prep_masa(experiment, collection)