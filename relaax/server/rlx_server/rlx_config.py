import logging
import re

from relaax.common.python.config.base_config import BaseConfig


logger = logging.getLogger(__name__)


class RLXConfig(BaseConfig):

    def __init__(self):
        super(RLXConfig, self).__init__()

    def load_from_cmdl(self, parser):
        add = parser.add_argument
        add('--config', type=str, default=None, required=True,
            help='RELAAX config yaml file')
        add('--bind', type=str, default=None,
            help='Address of the RLX server in the format host:port')
        add('--metrics-server', type=str, default=None,
            help=('Address of the metrics server '
                  'address in the format host:port'))
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

        if self.metrics_server is None:
            self.metrics_server = self.get(
                'relaax_metrics_server/bind', 'localhost:7002')

        if self.parameter_server is None:
            self.parameter_server = self.get(
                'relaax_parameter_server/bind', 'localhost:7000')

        self.protocol_name = self.get(
            'relaax_rlx_server/protocol', 'rawsocket')

        self.algorithm_path = self.get('algorithm/path')
        self.algorithm_name = self.get('algorithm/name')

        # Simple check of the servers addresses format
        self.bind = self.parse_address(self.bind, 'RLX server bind')
        self.metrics_server = self.parse_address(self.metrics_server, 'metrics server')
        self.parameter_server = self.parse_address(self.parameter_server, 'parameter server')

    def parse_address(self, address, address_name):
        m = re.match('^\s*([^\s:]*)\s*:\s*(\d+)\s*$', address)
        if m is None:
            logger.error('Please specify %s address in host:port format.', address_name)
            exit(1)
        return m.group(1), int(m.group(2))


options = RLXConfig().load()
