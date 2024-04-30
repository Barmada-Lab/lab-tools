import logging

from trogon import tui
import click

from cytomancer.cvat.survival import cli_entry as cvat_survival
from cytomancer.cvat.upload import cli_entry_basic, cli_entry_experiment
from cytomancer.cvat.measure import cli_entry as cvat_measure
from cytomancer.cvat.nuc_cyto import cli_entry as cvat_nuc_cyto

from cytomancer.settings import settings

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
