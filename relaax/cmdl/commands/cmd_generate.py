from __future__ import print_function
from __future__ import absolute_import
from builtins import object

import os
import errno
import shutil
import click

from relaax.cmdl.cmdl import pass_context
from relaax.cmdl.cmdl import ALGORITHMS, ALGORITHMS_META, ENVIRONMENTS, ENVIRONMENTS_META
from relaax.cmdl.utils.backup import Backup
from relaax.cmdl.commands.cmd_config import CmdConfig


class CmdGenerate(object):

    def __init__(self, ctx, app_path, algorithm, environment, copy_algorithm=True, create_config_backup=True):
        self.ctx = ctx
        self.app_path = app_path
        self.algorithm = algorithm
        self.environment = environment
        self.copy_algorithm = copy_algorithm
        self.create_config_backup = create_config_backup

    def apply(self):
        if (self.algorithm is not None) and self.copy_algorithm:
            self._generate('%salgorithm' % self.app_path, self.algorithm, 'algorithm',
                           '../../algorithms/%s' % self.algorithm.replace("-", "_"))
        if self.environment is not None:
            self._generate('%senvironment' % self.app_path, self.environment, 'environment',
                           '../../templates/environments/%s/environment' % self.environment)

        CmdConfig(self.ctx, '%sapp.yaml' % self.app_path, self.algorithm, self.environment,
                  set_algorithm_path=self.copy_algorithm, create_backup=self.create_config_backup).apply()

    def _generate(self, target_path, template_name, template_type, template_path):
        backup_created, backup_name = Backup(target_path).make_backup()
        if backup_created:
            self.ctx.log('Created backup of the previous %s in the %s folder' % (template_type, backup_name))

        module_path = os.path.dirname(os.path.abspath(__file__))
        template_path = os.path.abspath(os.path.join(module_path, template_path))

        try:
            shutil.copytree(template_path, target_path)
        except OSError as e:
            if e.errno == errno.ENOENT:
                raise Exception(
                    'Can\'t create \'%s\'. Base %s doesn\'t exist.' % (template_name, template_type))
            raise

        if template_type == 'environment':
            fn = os.path.join(self.app_path, 'README.md')
            os.remove(fn) if os.path.exists(fn) else None
            try:
                shutil.copy2(os.path.abspath(os.path.join(template_path, '../README.md')), self.app_path)
            except:
                self.ctx.log('Can\'t find README for specified environment')

        self.ctx.log('Created %s %s in %s' % (template_name, template_type, target_path))

    def create_default_config(self, environment):
        module_path = os.path.dirname(os.path.abspath(__file__))
        default_config = os.path.abspath(os.path.join(
            module_path, '../../templates/environments/%s/app.yaml' % environment))
        shutil.copy2(default_config, self.app_path)


@click.command('generate', short_help='Generate parts of RELAAX application.')
@click.option('--algorithm', '-a', default=None, metavar='',
              type=click.Choice(ALGORITHMS),
              help='[%s]\nGenerate RELAAX algorithm in the app folder. '
              'You may use "relaax config" to configure algorithm for the '
              'specific environment.' % ALGORITHMS_META)
@click.option('--environment', '-e', default=None, metavar='',
              type=click.Choice(ENVIRONMENTS),
              help='[%s] \n Generate environment.' % ENVIRONMENTS_META)
@pass_context
def cmdl(ctx, algorithm, environment):
    """Generate parts of RELAAX application.

    Generate specified algorithm implementation or/and
    RL environment based on specified framework.
    """
    ctx.setup_logger(format='')
    CmdGenerate(ctx, './', algorithm, environment).apply()
