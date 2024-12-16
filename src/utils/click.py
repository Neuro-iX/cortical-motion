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
