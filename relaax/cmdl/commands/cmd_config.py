from __future__ import print_function
from builtins import str
from builtins import object

import os
import sys
import click
import ruamel.yaml

from relaax.cmdl.cmdl import pass_context
from relaax.cmdl.cmdl import ALGORITHMS, ALGORITHMS_META, ENVIRONMENTS, ENVIRONMENTS_META
from relaax.cmdl.utils.backup import Backup


class CmdConfig(object):

    def __init__(self, ctx, config, algorithm, environment, create_backup=True, set_algorithm_path=True):
        self.ctx = ctx
        self.config = config
        self.algorithm = algorithm
        self.environment = environment
        self.create_backup = create_backup
        self.set_algorithm_path = set_algorithm_path

    def apply(self):
        if self.algorithm is None and self.environment is None:
            self.ctx.log('Nothing to do, exiting...')
            return
        if self.algorithm is not None:
            self._configure(self.algorithm,
                            'algorithm',
                            '../../templates/configs/%s.yaml' % self.algorithm,
                            self.set_algorithm_path)
        if self.environment is not None:
            self._configure(self.environment,
                            'environment',
                            '../../templates/environments/%s/app.yaml' % self.environment)

    def _configure(self, template_name, block_name, template_path, set_algorithm_path=False):
        app_config = self._read_yaml(self.config, 'Sorry, can\'t read input yaml: %s')

        module_path = os.path.dirname(os.path.abspath(__file__))
        template_config_path = os.path.abspath(os.path.join(
            module_path, template_path))

        template_config = self._read_yaml(template_config_path, 'Sorry, can\'t read config template: %s')

        app_config[block_name] = template_config[block_name]

        if set_algorithm_path:
            app_config[block_name].insert(0, 'path', './algorithm',
                                          'folder to load algorithm from; delete this key '
                                          'to use default implementation of the algorithm')

        if self.create_backup:
            backup_created, backup_name = Backup(self.config).make_backup()
            if backup_created:
                self.ctx.log('Created backup of the previous %s in the %s' %
                             (os.path.basename(self.config), backup_name))
            self.create_backup = False

        self._write_yaml(self.config, app_config)

        self.ctx.log('Configured %s with %s %s default parameters.' %
                     (os.path.basename(self.config), template_name, block_name))

    def _write_yaml(self, filename, config):
        try:
            with open(filename, 'w') as out:
                out.write(ruamel.yaml.dump(config, Dumper=ruamel.yaml.RoundTripDumper))
        except EnvironmentError as e:   # parent of IOError, OSError *and* WindowsError where available
            self.ctx.log('Sorry, can\'t write config: %s' % str(e))
            sys.exit()

    def _read_yaml(self, filename, error_message):
        try:
            with open(filename, 'r') as inp:
                yaml = ruamel.yaml.load(inp, ruamel.yaml.RoundTripLoader)
        except EnvironmentError as e:   # parent of IOError, OSError *and* WindowsError where available
            self.ctx.log(error_message % str(e))
            sys.exit()
        return yaml

    @staticmethod
    def list_configurations():
        module_path = os.path.dirname(os.path.abspath(__file__))
        template_config_path = os.path.abspath(os.path.join(
            module_path, '../../templates/configs'))

        rv = []
        for filename in os.listdir(template_config_path):
            if filename.endswith('.yaml'):
                rv.append(filename[:-5])
        rv.sort()
        return (rv, '|'.join(rv))


CONFIGURATIONS = CmdConfig.list_configurations()


@click.command('config', short_help='Configure algorithm.')
@click.option('--config', '-c', type=click.File(lazy=True), show_default=True, default='app.yaml',
              help='Relaax configuraion yaml file.')
@click.option('--algorithm', '-a', default=None, metavar='',
              type=click.Choice(CONFIGURATIONS[0]),
              help='[%s]\nRelaax algorithm configuration.' % CONFIGURATIONS[1])
@click.option('--environment', '-e', default=None, metavar='',
              type=click.Choice(ENVIRONMENTS),
              help='[%s] \n Environment configuration for the application.' % ENVIRONMENTS_META)
@click.option('--local-algorithm', '-l', default=False, type=bool, show_default=True,
              help='Set this flag to True if your algorithm implementation is in app folder. '
              'algoritm/path key will be added to the config yaml file and algorithm will be load '
              'from that path.')
@pass_context
def cmdl(ctx, config, algorithm, environment, local_algorithm):
    """Configure Relaax algorithm.

    Fill algorithm or/and environment section of the configuration file with
    default parameters for the specified algorithm or/and environment.
    Old configuration file will be copied to a backup file.
    """
    ctx.setup_logger(format='')
    CmdConfig(ctx, config.name, algorithm=algorithm, environment=environment,
              create_backup=True, set_algorithm_path=local_algorithm).apply()
