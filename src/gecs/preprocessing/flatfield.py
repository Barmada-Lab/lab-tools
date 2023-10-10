from pathlib import Path

import click

from improc.experiment.types import Experiment
from improc.experiment import loader
from improc.processes.types import Pipeline
from improc.processes.flatfield import BaSiC

def flatfield(experiment: Experiment, collection: str, group_by: list[str], parallelism: int = 4):
    pipeline = Pipeline(
        BaSiC(group_by=group_by, parallelism=parallelism))
    pipeline.run(experiment, collection)

@click.command("flatfield")
@click.argument('experiment_dir', type=click.Path(exists=True, path_type=Path))
@click.option('--scratch-dir', type=click.Path(path_type=Path), default=None, help="Path to experiment directory")
@click.option('--collection', type=str, default="raw_imgs", )
@click.option('--group-by', type=str, multiple=True, default=['vertex', 'mosaic', 'exposure'])
def cli_entry(args):
    """ Performs BaSiC illumination correction on groups of raw images. """
    scratch_dir = args.scratch_dir if args.scratch_dir is not None else args.experiment_dir / "processed_imgs"
    experiment = loader.load_experiment(args.experiment_dir, scratch_dir)
    flatfield(experiment, args.collection, args.group_by, args.parallelism)