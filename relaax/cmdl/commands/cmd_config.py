from __future__ import print_function
from builtins import object

import os
import click

from relaax.cmdl.cmdl import pass_context
from relaax.cmdl.cmdl import ENVIRONMENTS, ENVIRONMENTS_META
from relaax.cmdl.utils.backup import Backup
from relaax.cmdl.utils.commented_yaml import CommentedYaml


SPECIAL_CONFIG_LIST = {
    'da3c-openai-gym': 'da3c-cartpole',
    'dppo-bandit': 'dppo-bandit',
    'dppo-openai-gym': 'dppo-cartpole',
    'dqn-bandit': 'dqn-bandit',
    'dqn-openai-gym': 'dqn-cartpole',
    'policy-gradient-bandit': 'pg-bandit',
    'policy-gradient-openai-gym': 'pg-cartpole',
    'trpo-bandit': 'trpo-bandit'
}


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
            CmdConfig.list_configurations(self.ctx)
        else:
            if self.environment is not None:
                self._configure(self.environment,
                                'environment',
                                '../../templates/environments/%s/app.yaml' % self.environment)
            if self.algorithm is not None:
                self._configure(self.algorithm,
                                'algorithm',
                                '../../templates/configs/%s.yaml' % self.algorithm,
                                self.set_algorithm_path)

    def _configure(self, template_name, block_name, template_path, set_algorithm_path=False):
        app_config = CommentedYaml.read(
            self.ctx, self.config, 'Sorry, can\'t read input yaml: %s')
        template_config = CommentedYaml.read(
            self.ctx, self.template_config_path(template_path), 'Sorry, can\'t read config template: %s')

        if app_config['algorithm']['name'] != template_config['algorithm']['name']:

            config_name = SPECIAL_CONFIG_LIST.get(self.algorithm + '-' + self.environment)
            if config_name is not None:
                template_config = CommentedYaml.read(self.ctx, self.template_config_path(
                    '../../templates/configs/%s.yaml' % config_name),
                    'Sorry, can\'t read special config template: %s')

            try:
                env_section = template_config['environment']
                app_config['environment'] = template_config['environment']
                if env_section['type'].lower() != self.environment:
                    self.ctx.log('WARNING: template for %s algorithm configured for %s %s environment',
                                 self.algorithm, env_section['type'], env_section['name'])
                    self.ctx.log('WARNING: this is NOT equal to requested %s environment type',
                                 self.environment)
            except Exception:
                pass

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

        CommentedYaml.write(self.ctx, self.config, app_config)

        self.ctx.log('Configured %s with %s %s default parameters.' %
                     (os.path.basename(self.config), template_name, block_name))

    @staticmethod
    def template_config_path(template):
        module_path = os.path.dirname(os.path.abspath(__file__))
        return os.path.abspath(os.path.join(module_path, template))

    @staticmethod
    def configurations():
        rv = []
        for filename in os.listdir(CmdConfig.template_config_path('../../templates/configs')):
            if filename.endswith('.yaml'):
                rv.append(filename[:-5])
        rv.sort()
        return rv, '|'.join(rv)

    @staticmethod
    def list_configurations(ctx):
        ctx.log('Available algorithm configurations:\n')
        templates = CmdConfig.template_config_path('../../templates/configs')
        for filename in os.listdir(templates):
            if filename.endswith('.yaml'):
                ctx.log(click.style(filename[:-5], bold=True))
                config = CommentedYaml.read(
                    ctx, os.path.join(templates, filename), 'Sorry, can\'t read %s' % filename)
                try:
                    for comment_tokens in config.ca.comment:
                        if comment_tokens:
                            output = [comment.value for comment in comment_tokens]
                            ctx.log(''.join(output).rstrip())
                except Exception:
                    ctx.log('No description provided')
        ctx.log('\nRun "relaax config --help" to see how to apply these configurations.\n')


CONFIGURATIONS = CmdConfig.configurations()


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

    Run "relaax config" without any options to see all available algorithm configurations.
    """
    ctx.setup_logger(format='')
    CmdConfig(ctx, config.name, algorithm=algorithm, environment=environment,
              create_backup=True, set_algorithm_path=local_algorithm).apply()
