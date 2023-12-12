import click
from trogon import tui

from .analysis.survival import cli_entry as survival

from .cvat.survival import cli_entry as cvat_survival
from .cvat.upload import cli_entry as cvat_upload
from .cvat.measure import cli_entry as cvat_measure
from .cvat.nuc_cyto import cli_entry as cvat_nuc_cyto

@tui()
@click.group()
@click.pass_context
def cli(ctx):
    ctx.ensure_object(dict)

@cli.group()
def analyze():
    pass

analyze.add_command(survival)

@cli.group()
def cvat():
    pass

cvat.add_command(cvat_survival)
cvat.add_command(cvat_upload)
cvat.add_command(cvat_measure)
cvat.add_command(cvat_nuc_cyto)
