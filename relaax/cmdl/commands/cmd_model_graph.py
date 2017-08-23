from builtins import object

import click
import honcho.manager
import os
import sys
import webbrowser

from relaax.server.common import algorithm_loader
from relaax.common.python.config.config_yaml import ConfigYaml
from relaax.cmdl.cmdl import pass_context

         
class CmdModelGraph(object):

    def __init__(self, ctx, config):
        self.config = config
        self.config_yaml = ConfigYaml()
        self.config_yaml.load_from_file(self.config)

    def show_graph(self):
        sys.exit(self.run_tensorboard())

    def run_tensorboard(self):

        for p, n in algorithm_loader.AlgorithmLoader.model_packages(self.config_yaml.get('algorithm/path'),
                                                                    self.config_yaml.get('algorithm/name')):
            os.system('python -m %s --config %s' % (n, self.config))
        webbrowser.open('http://localhost:6006', new=2)  # open in new tab
        os.system('tensorboard --logdir %s' % 'log')


@click.command('model_graph', short_help='Show TensorFlow graph.')
@click.option('--config', '-c', type=click.File(lazy=True), show_default=True, default='app.yaml',
              help='Relaax configuraion yaml file.')
@pass_context
def cmdl(ctx, config):
    """Shows TensorFlow graph.

    For example:
        $relaax model_graph
    """
    ctx.setup_logger(format='%(asctime)s %(name)s\t\t  | %(message)s')
    # Disable TF warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # Execute command
    CmdModelGraph(ctx, config.name).show_graph()
