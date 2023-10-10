import pathlib as pl

import click

from survival import core_replacement

@click.command("survival")
@click.argument("experiment_base", type=click.Path(path_type=pl.Path, exists=True))
@click.argument("scratch", type=click.Path(path_type=pl.Path, exists=True))
def cli_entry(experiment_base: pl.Path, scratch: pl.Path):
    core_replacement.run(experiment_base, scratch)