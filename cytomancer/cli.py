from pathlib import Path
import logging

from trogon import tui
import click

from cytomancer.utils import experiment_path_argument, test_cvat_connection
from cytomancer.experiment import ExperimentType
from cytomancer.cvat.survival import cli_entry as cvat_survival
from cytomancer.cvat.upload import cli_entry_experiment
from cytomancer.cvat.nuc_cyto import cli_entry as cvat_nuc_cyto

from cytomancer.updater import check_for_updates
from cytomancer.config import settings

logging.basicConfig(level=settings.log_level)


@tui()
@click.group(help="Cytomancer CLI")
@click.pass_context
def cli(ctx):
    check_for_updates()
    ctx.ensure_object(dict)


# ----------------- BEGIN QUANT GROUP -----------------

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
def quant_group(ctx):
    ctx.ensure_object(dict)


quant_group.add_command(pultra_survival)


# ----------------- BEGIN CVAT GROUP -----------------

@cli.group("cvat", help="Tools for working with CVAT")
@click.pass_context
def cvat_group(ctx):
    ctx.ensure_object(dict)


@click.command("auth")
@click.option("--cvat-username", prompt="CVAT Username")
@click.password_option("--cvat-password", prompt="CVAT Password")
def cvat_auth(cvat_username, cvat_password):
    """
    Update CVAT credentials. Run this with no arguments to get an interactive prompt that hides your password.
    """

    print(f"\nTesting CVAT connection to server {settings.cvat_url}...")
    if not test_cvat_connection(settings.cvat_url, cvat_username, cvat_password):
        print("Connection failed. Please verify your credentials and try again.")
        print("See `cyto config update --help` for other CVAT-related settings")
        return

    print("Authentication successful. Saving credentials.")
    settings.cvat_username = cvat_username
    settings.cvat_password = cvat_password
    settings.save()


cvat_group.add_command(cvat_auth)
cvat_group.add_command(cvat_survival)
cvat_group.add_command(cli_entry_experiment)
cvat_group.add_command(cvat_nuc_cyto)


# ----------------- BEGIN SETTINGS GROUP -----------------

@cli.group("config", help="Config management")
@click.pass_context
def config_group(ctx):
    ctx.ensure_object(dict)


@click.command("show")
def show_config():
    """
    Display current configuration settings.
    """
    print("\nCurrent settings:")
    for k, v in settings.model_dump().items():
        if k == "cvat_password":
            v = "*" * len(v)
        print(f"\t{k}: {v}")
    print()


# Hacky little thing that adds options for all settings to set_config
def settings_options():
    def combined_decorator(func):
        for k, v in reversed(settings.model_dump().items()):
            if k == "cvat_password" or k == "cvat_username":
                continue
            decorator = click.option(f"--{k}", default=v, show_default=True)
            func = decorator(func)
        return func
    return combined_decorator


@click.command("update")
@settings_options()
def update_config(**kwargs):
    """
    Update config
    """
    for k, v in kwargs.items():
        setattr(settings, k, v)
    settings.save()


config_group.add_command(show_config)
config_group.add_command(update_config)
