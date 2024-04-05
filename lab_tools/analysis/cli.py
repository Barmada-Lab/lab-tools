from trogon import tui
import click

from lab_tools.analysis.pultra_survival import cli_entry as nuc_survival


@tui()
@click.group()
@click.pass_context
def cli(ctx):
    ctx.ensure_object(dict)


cli.add_command(nuc_survival)
