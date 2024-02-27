import logging

from trogon import tui
import click

from .analysis.survival import cli_entry as survival

from .cvat.survival import cli_entry as cvat_survival
from .cvat.upload import cli_entry_basic, cli_entry_experiment
from .cvat.measure import cli_entry as cvat_measure
from .cvat.nuc_cyto import cli_entry as cvat_nuc_cyto
from .models.nuclei_survival_svm import cli_entry as nuclei_survival_svm

from gecs.settings import settings

logging.basicConfig(level=settings.log_level)


@tui()
@click.group()
@click.pass_context
def cli(ctx):
    ctx.ensure_object(dict)


@cli.group()
def analyze():
    pass


analyze.add_command(survival)
analyze.add_command(nuclei_survival_svm)


@cli.group()
def cvat():
    pass


cvat.add_command(cvat_survival)
cvat.add_command(cli_entry_basic)
cvat.add_command(cli_entry_experiment)
cvat.add_command(cvat_measure)
cvat.add_command(cvat_nuc_cyto)
