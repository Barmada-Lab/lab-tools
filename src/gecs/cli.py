from pathlib import Path

import click
from trogon import tui

from .preprocessing.composite import cli_entry as composite
from .preprocessing.sns import cli_entry as sns
from .preprocessing.flatfield import cli_entry as flatfield
from .preprocessing.project import cli_entry as mip

from .analysis.measure import cli_entry as measure
from .analysis.colocalize import cli_entry as correlate

from .util.masa_prep import cli_entry as masa_prep
from .util.cvat_deploy import cli_entry as cvat_deploy

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

analyze.add_command(measure)
analyze.add_command(correlate)

@cli.group()
def util():
    pass

util.add_command(masa_prep)
util.add_command(cvat_deploy)
