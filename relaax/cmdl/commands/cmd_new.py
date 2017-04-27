import os
import errno
import click
import shutil

from ..cmdl import pass_context


class NewApp(object):

    def __init__(self, ctx, app_name, environment):
        self.ctx = ctx
        self.app_name = app_name
        self.base_env = environment

    def mk_environment(self):
        app_path = os.path.abspath(os.path.join(os.getcwd(), self.app_name))
        module_path = os.path.dirname(os.path.abspath(__file__))
        template_path = os.path.abspath(os.path.join(
            module_path, '../../templates/environments/%s' % self.base_env))
        try:
            shutil.copytree(template_path, app_path)
        except OSError as e:
            if e.errno == errno.EEXIST:
                raise Exception('Can\'t create \'%s\'. Folder already exists.' % self.app_name)
            elif e.errno == errno.ENOENT:
                raise Exception('Can\'t create \'%s\'. Base environment doesn\'t exist.' % self.app_name)
            raise

    def create(self):
        try:
            self.mk_environment()
            self.ctx.log('Created application \'%s\' based on \'%s\' environment',
                         self.app_name, self.base_env)
            self.ctx.log('To run application, please do: cd %s && relaax run' % self.app_name)
        except Exception as e:
            self.ctx.log('%s', str(e))


@click.command('new', short_help='Create new RELAAX application.')
@click.argument('app-name', required=True, type=click.STRING)
@click.option('--environment', '-e', default='basic', show_default=True, metavar='',
              type=click.Choice(['basic', 'openai-gym', 'deepmind-lab']),
              help='[basic|openai-gym|deepmind-lab] \n Environment to base application on.')
@pass_context
def cmdl(ctx, app_name, environment):
    """Build new RELAAX application."""
    ctx.setup_logger()
    NewApp(ctx, app_name, environment).create()
