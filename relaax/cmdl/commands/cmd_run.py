from builtins import object
import os
import sys
import click
from honcho.manager import Manager

from relaax.common.python.config.config_yaml import ConfigYaml
from relaax.cmdl.cmdl import pass_context


class RManager(Manager):

    def __init__(self, *args, **kwargs):
        super(RManager, self).__init__(*args, **kwargs)

    def _any_stopped(self):
        clients = []
        for _, p in self._processes.items():
            if p['process'].name.startswith('environment'):
                clients.append(p.get('returncode') is not None)
        if len(clients):
            return all(clients)
        else:
            super(RManager, self)._any_stopped()


class CmdlRun(object):

    def __init__(self, ctx, components, config, n_clients, exploit, show_ui):
        self.ctx = ctx
        self.config = config
        self.show_ui = show_ui
        self.exploit = exploit
        self.n_clients = n_clients
        self.components = components if bool(components) else set(['all'])

        if sys.platform == 'win32':
            self.nobuffer = ''
        else:
            self.nobuffer = 'PYTHONUNBUFFERED=true'
        self.config_yaml = ConfigYaml()
        self.config_yaml.load_from_file(self.config)

    def run_componenets(self):

        manager = RManager()

        self.run_parameter_server(manager)
        self.run_rlx_server(manager)
        self.run_wsproxy(manager)
        self.run_client(manager)

        manager.loop()
        sys.exit(manager.returncode)

    def intersection(self, lst):
        return bool(self.components.intersection(lst))

    def isconfigured(self, server):
        return self.config_yaml.get(server, None) is not None

    def run_parameter_server(self, manager):
        if self.intersection(['all', 'servers', 'parameter-server']):
            if self.isconfigured('relaax_parameter_server'):
                manager.add_process('parameter-server',
                                    '%s relaax-parameter-server --config %s'
                                    % (self.nobuffer, self.config))
            else:
                self.ctx.log(click.style("parameter-server is not configured", fg='red'))

    def run_rlx_server(self, manager):
        if self.intersection(['all', 'servers', 'rlx-server']):
            if self.isconfigured('relaax_rlx_server'):
                manager.add_process('rlx-server',
                                    '%s relaax-rlx-server --config %s'
                                    % (self.nobuffer, self.config))
            else:
                self.ctx.log(click.style("rlx-server is not configured", fg='red'))

    def run_wsproxy(self, manager):
        if self.intersection(['all', 'servers', 'wsproxy']):
            if self.isconfigured('relaax_wsproxy'):
                manager.add_process('wsproxy',
                                    '%s relaax-wsproxy --config %s'
                                    % (self.nobuffer, self.config))
            elif self.intersection(['wsproxy']):
                # log error message only if wsproxy specified explicitly
                self.ctx.log(click.style("wsproxy is not configured", fg='red'))

    def run_client(self, manager):
        if self.intersection(['all', 'environment']):
            self.client = self.config_yaml.get('environment/run')
            if self.client:
                self.run_all_clients(manager)
            else:
                self.ctx.log(click.style("environment is not configured", fg='red'))

    def run_all_clients(self, manager):
        count = 0
        while count < self.n_clients:
            if count == 0:
                self.run_one_client('environment-%d' % count, manager, self.exploit, self.show_ui)
            else:
                self.run_one_client('environment-%d' % count, manager)
            count += 1

    def run_one_client(self, process_name, manager, exploit=False, show_ui=False):
        manager.add_process(
            process_name, '%s %s --config %s --exploit %s --show-ui %s' %
            (self.nobuffer, self.client, self.config, exploit, show_ui))


@click.command('run', short_help='Run RELAAX components.')
@click.argument('components', nargs=-1, type=click.Choice(
                ['all', 'environment', 'servers', 'rlx-server', 'parameter-server', 'wsproxy']))
@click.option('--config', '-c', type=click.File(lazy=True), show_default=True, default='app.yaml',
              help='Relaax configuraion yaml file.')
@click.option('--n-environments', '-n', default=1, show_default=True,
              help='Number of environments to run at the same time.')
@click.option('--exploit', default=False, type=bool, show_default=True,
              help='Only first started environment will get provided exploit flag in command line parameters.'
              ' Rest of the started environments will get exploit flag set to False.')
@click.option('--show-ui', is_flag=True, show_default=True,
              help='Only first started environment will get show-ui flag in command line parameters. '
              'Rest of the started environments will get show-ui flag set to False.')
@pass_context
def cmdl(ctx, components, config, n_environments, exploit, show_ui):
    """Run RELAAX components.

    \b
    COMPONENTS:
    all              - run environments and servers (default)
    environment      - run environment
    servers          - run rlx-server, parameter-server, and wsproxy (if specified in config yaml)
    rlx-server       - run rlx-server
    parameter-server - run parameter-server
    wsproxy          - run websockets proxy

    \b
    For example:
        - run environment, rlx-server, parameter-server, and wsproxy
        $relaax run all
        - run rlx-server, parameter-server, and wsproxy
        $relaax run servers
    """
    ctx.setup_logger(format='%(asctime)s %(name)s\t\t  | %(message)s')
    # Disable TF warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # Exacute command
    CmdlRun(ctx, set(components), config.name, n_environments, exploit, show_ui).run_componenets()
