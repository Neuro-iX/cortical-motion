import click
import matplotlib

from src.process.process import process
from src.training.training import train
from src.utils.log import rich_logger

matplotlib.use("Agg")


@click.group
def cli():
    pass


if __name__ == "__main__":
    rich_logger()
    cli.add_command(process)
    cli.add_command(train)
    cli()
