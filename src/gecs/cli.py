from pathlib import Path

import click
from trogon import tui

from .preprocessing.composite import cli_entry as composite
from .preprocessing.sns import cli_entry as sns
from .preprocessing.flatfield import cli_entry as flatfield
from .preprocessing.project import cli_entry as mip

from .analysis.colocalize import cli_entry as correlate
from .analysis.survival import cli_entry as survival

from .util.cvat_survival_analysis import cli_entry as cvat_survival_analysis
from .util.cvat_upload_images import cli_entry as cvat_upload_images
from .util.cvat_upload_experiment import cli_entry as cvat_upload_experiment

@tui()
@click.group()
@click.pass_context
def cli(ctx):
    ctx.ensure_object(dict)

@cli.group()
def preprocess():
    pass

preprocess.add_command(composite)
preprocess.add_command(sns)
preprocess.add_command(flatfield)
preprocess.add_command(mip)

@cli.group()
def analyze():
    pass

analyze.add_command(correlate)
analyze.add_command(survival)

@cli.group()
def util():
    pass

util.add_command(cvat_survival_analysis)
util.add_command(cvat_upload_images)
util.add_command(cvat_upload_experiment)
