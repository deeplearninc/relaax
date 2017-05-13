import logging as log

from relaax.common.python.config.base_config import BaseConfig


class RLXConfig(BaseConfig):

    def __init__(self):
        super(RLXConfig, self).__init__()

    def load_from_cmdl(self, parser):
        add = parser.add_argument
        add('--config', type=str, default=None, required=True,
            help='RELAAX config yaml file')
        add('--bind', type=str, default=None,
            help='Address of the RLX server in the format host:port')
        add('--parameter-server', type=str, default=None,
            help=('Address of the parameter server '
                  'address in the format host:port'))
        add('--log-level', type=str, default=None,
            choices=('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'),
            help='Logging level')
        add('--short-log-messages', type=bool, default=True,
            help='Log messages will skip long log prefix')
        add('--timeout', type=float, default=None,
            help='Force Agent(s) to stops and restart after given timeout')

    def process_after_loaded(self):
        if self.log_level is None:
            self.log_level = self.get(
                'relaax_rlx_server/log_level', 'DEBUG')

        if self.bind is None:
            self.bind = self.get(
                'relaax_rlx_server/bind', 'localhost:7001')

        if self.timeout is None:
            self.timeout = self.get(
                'relaax-rlx-server/--timeout', 1000000)

        if self.parameter_server is None:
            self.parameter_server = self.get(
                'relaax_parameter_server/bind', 'localhost:9000')

        self.protocol_name = self.get(
            'relaax_rlx_server/protocol', 'rawsocket')

        self.algorithm_path = self.get('algorithm/path')
        self.algorithm_name = self.get('algorithm/name')

        # Simple check of the servers addresses format

        self.bind = [x.strip() for x in self.bind.split(':')]
        if len(self.bind) != 2:
            log.error(
                "Please specify RLX server address in host:port format")
            exit()
        self.bind[1] = int(self.bind[1])
        self.bind = tuple(self.bind)

        self.parameter_server = [x.strip() for x in self.parameter_server.split(':')]
        if len(self.parameter_server) != 2:
            log.error(
                ('Please specify Parameter Server '
                 'server address in host:port format'))
            exit()
        self.parameter_server[1] = int(self.parameter_server[1])
        self.parameter_server = tuple(self.parameter_server)


options = RLXConfig().load()
