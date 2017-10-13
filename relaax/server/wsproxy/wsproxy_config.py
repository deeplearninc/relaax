from __future__ import print_function

import logging as log
from relaax.common.python.config.base_config import BaseConfig


class WSProxyConfig(BaseConfig):

    def __init__(self):
        super(BaseConfig, self).__init__()

    def load_from_cmdl(self, parser):
        add = parser.add_argument
        add('--config', type=str, default=None, required=True,
            help='RELAAX config yaml file')
        add('--bind', type=str, default=None,
            help='Address of the RLX server in the format host:port')
        add('--rlx-server', type=str, default=None,
            help=('Address of the RLX server '
                  'address in the format host:port'))
        add('--log-level', type=str, default=None,
            choices=('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'),
            help='Logging level')
        add('--short-log-messages', type=bool, default=True,
            help='Log messages will skip long log prefix')

    def process_after_loaded(self):
        if self.log_level is None:
            self.log_level = self.get('relaax_wsproxy/log_level', 'DEBUG')

        if self.bind is None:
            self.bind = self.get('relaax_wsproxy/bind', 'localhost:7002')

        if self.rlx_server is None:
            self.rlx_server = self.get(
                'relaax_rlx_server/bind', 'localhost:7001')

        # Simple check of the servers addresses format

        self.bind = [x.strip() for x in self.bind.split(':')]
        if len(self.bind) != 2:
            log.error('Please specify wsproxy server address in host:port format')
            exit()
        self.bind[1] = int(self.bind[1])
        self.bind = tuple(self.bind)

        self.rlx_server = [x.strip() for x in self.rlx_server.split(':')]
        if len(self.rlx_server) != 2:
            log.error('Please specify RLX Server server address in host:port format')
            exit()
        self.rlx_server[1] = int(self.rlx_server[1])
        self.rlx_server = tuple(self.rlx_server)


options = WSProxyConfig().load()
