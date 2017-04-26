import click
from ..cmdl import pass_context


@click.command('generate', short_help='Generate parts of RELAAX application.')
@click.argument('path', required=False, type=click.Path(resolve_path=True))
@pass_context
def cmdl(ctx, path):
    """Generate parts of RELAAX application."""
    pass
