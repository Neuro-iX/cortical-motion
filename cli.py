import click

from src.process.commands import process
from src.utils.log import rich_logger


@click.group
def cli():
    pass


if __name__ == "__main__":
    rich_logger()
    cli.add_command(process)
    cli()
