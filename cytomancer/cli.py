from pathlib import Path
import logging

from trogon import tui
import click

from cytomancer.utils import experiment_path_argument
from cytomancer.experiment import ExperimentType
from cytomancer.cvat.survival import cli_entry as cvat_survival
from cytomancer.cvat.upload import cli_entry_experiment
from cytomancer.cvat.nuc_cyto import cli_entry as cvat_nuc_cyto

from cytomancer.updater import check_for_updates
from cytomancer.settings import settings

logging.basicConfig(level=settings.log_level)


@tui()
@click.group(help="Cytomancer CLI")
@click.pass_context
def cli(ctx):
    check_for_updates()
    ctx.ensure_object(dict)


@click.command("pult-surv")
@experiment_path_argument()
@click.option("--save-annotations", is_flag=True, help="Save annotated stacks to results folder")
@click.option("--run-sync", is_flag=True, help="Run synchronously, skipping the task queue.")
def pultra_survival(experiment_path: Path, save_annotations: bool, sync: bool):
    """
    Run pultra survival analysis on an experiment. Note that only CQ1 acquisitions are supported.
    """

    if sync:
        from cytomancer.quant.pultra_survival import run
        run(experiment_path, ExperimentType.CQ1, save_annotations)
    else:
        from cytomancer.quant.tasks import run_pultra_survival
        run_pultra_survival.delay(str(experiment_path), ExperimentType.CQ1, save_annotations)


@cli.group("quant", help="Tools for quantifying data")
@click.pass_context
def quant(ctx):
    ctx.ensure_object(dict)


quant.add_command(pultra_survival)


@cli.group("cvat", help="Tools for working with CVAT")
@click.pass_context
def cvat(ctx):
    ctx.ensure_object(dict)


cvat.add_command(cvat_survival)
cvat.add_command(cli_entry_experiment)
cvat.add_command(cvat_nuc_cyto)
