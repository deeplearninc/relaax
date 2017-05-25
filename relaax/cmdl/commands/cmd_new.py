from __future__ import print_function
from builtins import str
from builtins import object

import os
import errno
import click

from ..cmdl import pass_context
from ..cmdl import ALGORITHMS, ALGORITHMS_META, ENVIRONMENTS, ENVIRONMENTS_META
from .cmd_generate import CmdGenerate


class NewApp(object):

    def __init__(self, ctx, app_name, algorithm, environment):
        self.ctx = ctx
        self.app_name = app_name
        self.algorithm = algorithm
        self.environment = environment

    def mk_app_folder(self):
        app_path = os.path.abspath(os.path.join(os.getcwd(), self.app_name))
        try:
            os.makedirs(app_path)
        except OSError as e:
            if e.errno == errno.EEXIST:
                raise Exception('Can\'t create \'%s\'. Folder already exists.' % self.app_name)
            raise
        self.ctx.log('Created application folder %s', self.app_name)
        return app_path

    def create(self):
        try:
            app_path = self.mk_app_folder()
            cmd_genereate = CmdGenerate(self.ctx, app_path + '/', self.algorithm, self.environment,
                                        copy_algorithm=False, create_config_backup=False)
            cmd_genereate.create_default_config()
            cmd_genereate.apply()

            if self.environment == 'openai-gym':
                self.ctx.log('Please make sure you have OpenAI Gym installed; '
                             'see installation instruction here: https://github.com/openai/gym')

            self.ctx.log('To run application, please do: cd %s && relaax run' % self.app_name)

        except Exception as e:
            self.ctx.log('%s', str(e))


@click.command('new', short_help='Create new RELAAX application.')
@click.argument('app-name', required=True, type=click.STRING)
@click.option('--algorithm', '-a', default='policy-gradient', show_default=True, metavar='',
              type=click.Choice(ALGORITHMS),
              help='[%s]\nAlgorithm to use with this application.' % ALGORITHMS_META)
@click.option('--environment', '-e', default='basic', show_default=True, metavar='',
              type=click.Choice(ENVIRONMENTS),
              help='[%s] \n Environment to base application on.' % ENVIRONMENTS_META)
@pass_context
def cmdl(ctx, app_name, algorithm, environment):
    """Create new RELAAX application."""
    ctx.setup_logger(format='')
    NewApp(ctx, app_name, algorithm, environment).create()
