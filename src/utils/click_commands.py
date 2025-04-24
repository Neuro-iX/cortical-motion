"""utilty function and type definition for click usage"""

from typing import Any, Callable

import click


def get_command_path(ctx: click.Context, func: Callable) -> tuple[str, ...]:
    """Find the command path, in the click Context, that calls a function
    It is used to replicate the received command and execute it in an other environment
    such as SLURM
    Args:
        ctx (click.Context): Context given to a command wrapper
        func (Callable): Function to search

    Returns:
        tuple[str]: Succession of command / groups called to reach the function
    """
    top = ctx.find_root()
    queue = [top.command]
    root_name = top.info_name or top.command.name

    if root_name is None:
        raise click.ClickException(
            "Given Context does not seem to contain named command"
        )
    path_queue: list[tuple[str, ...]] = [(root_name,)]
    while len(queue) > 0:
        current = queue.pop(0)
        path = path_queue.pop(0)
        if (
            hasattr(current, "callback")
            and current.callback is not None
            and current.callback.__name__ == func.__name__
        ):
            return path
        if isinstance(current, click.Group):
            for cmd_name, cmd in current.commands.items():
                queue.append(cmd)
                path_queue.append(path + (cmd_name,))

    raise click.ClickException(
        f"Could not find a command path for function '{func.__name__}' in this context"
    )


def get_command(ctx: click.Context, func: Callable, **kwargs: dict[str, Any]) -> str:
    """Retrieve the command called for a given function and arguments

    Args:
        ctx (click.Context): Click's context given to the wrapper
        func (Callable): Function to find
        **kwargs : Arguments given to Click Command

    Returns:
        str: String reproduction of the input command
    """
    cmd_path = get_command_path(ctx, func)

    cmd = " ".join(cmd_path)

    for key, value in kwargs.items():
        cmd += f" --{key} {value}"
    return cmd


class TupleParamType(click.ParamType):
    """Parse Tuple command option"""

    name = "tuple"

    def convert(self, value, param, ctx):

        if not value:
            return ()
        try:
            return tuple(int(item) for item in value.split(","))
        except ValueError:
            self.fail(
                f"Invalid tuple: {value}. Must be comma-separated integers.", param, ctx
            )


class ClickEnumType(click.ParamType):
    """Parse Enum command option"""

    def __init__(self, enum_class):
        self.name = enum_class.__name__
        self.enum_class = enum_class

    def convert(self, value, param, ctx):
        try:
            return self.enum_class[value]
        except KeyError:
            self.fail(
                f"Invalid value '{value}'. Valid options: {[e.name for e in self.enum_class]}",
                param,
                ctx,
            )
