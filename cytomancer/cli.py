import logging

from trogon import tui
import click

from cytomancer.quant.click import register as register_quant
from cytomancer.cvat.click import register as register_cvat
from cytomancer.oneoffs.click import register as register_oneoffs
from cytomancer.config_click import register as register_config

logger = logging.getLogger(__name__)


@tui()
@click.group(help="Cytomancer CLI")
@click.pass_context
def cli(ctx):
    ctx.ensure_object(dict)


register_quant(cli)
register_cvat(cli)
register_oneoffs(cli)
register_config(cli)
