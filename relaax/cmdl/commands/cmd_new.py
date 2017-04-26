import os
import errno
import click
from ..cmdl import pass_context


class NewApp(object):

    def __init__(self, ctx, app_name):
        self.ctx = ctx
        self.app_name = app_name

    def mk_folder(self, name):
        try:
            os.makedirs(name)
        except OSError as e:
            if e.errno == errno.EEXIST:
                raise Exception('Can\'t create \'%s\'. Folder already exists.' % name)
            raise

    def mk_environment(self):
        pass

    def create(self):
        try:
            self.ctx.log('Creating: %s', self.app_name)
            self.mk_folder(self.app_name)
        except Exception as e:
            self.ctx.log('%s', str(e))


@click.command('new', short_help='Create new RELAAX application.')
@click.argument('app-name', required=True, type=click.STRING)
@pass_context
def cmdl(ctx, app_name):
    """Build new RELAAX application."""
    ctx.setup_logger()
    NewApp(ctx, app_name).create()
