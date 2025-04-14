import click


def get_command_path(ctx: click.Context, func: click.Command) -> tuple[str]:
    top = ctx.find_root()
    queue = [top.command]
    path_queue = [(top.info_name,)]
    while len(queue) > 0:
        current = queue.pop(0)
        path = path_queue.pop(0)
        if hasattr(current, "callback") and current.callback.__name__ == func.__name__:
            return path
        if isinstance(current, click.Group):
            for cmd_name, cmd in current.commands.items():
                queue.append(cmd)
                path_queue.append(path + (cmd_name,))

    return None


def get_command(
    ctx: click.Context, func: click.Command, **kwargs: dict[str, any]
) -> str:
    print(func)
    cmd_path = get_command_path(ctx, func)
    cmd = " ".join(cmd_path)

    for key, value in kwargs.items():
        cmd += f" --{key} {value}"
    return cmd


class TupleParamType(click.ParamType):
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
