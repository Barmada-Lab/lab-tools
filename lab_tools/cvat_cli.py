import logging

from trogon import tui
import click

from lab_tools.cvat.survival import cli_entry as cvat_survival
from lab_tools.cvat.upload import cli_entry_basic, cli_entry_experiment
from lab_tools.cvat.measure import cli_entry as cvat_measure
from lab_tools.cvat.nuc_cyto import cli_entry as cvat_nuc_cyto

from lab_tools.settings import settings

logging.basicConfig(level=settings.log_level)


@tui()
@click.group()
@click.pass_context
def cli(ctx):
    ctx.ensure_object(dict)


cli.add_command(cvat_survival)
cli.add_command(cli_entry_basic)
cli.add_command(cli_entry_experiment)
cli.add_command(cvat_measure)
cli.add_command(cvat_nuc_cyto)
