import logging

from trogon import tui
import click

from lab_tools.settings import settings
from lab_tools.analysis.nuclei_survival_svm import cli_entry as nuc_survival

logging.basicConfig(level=settings.log_level)


@tui()
@click.group()
@click.pass_context
def cli(ctx):
    ctx.ensure_object(dict)


cli.add_command(nuc_survival)
