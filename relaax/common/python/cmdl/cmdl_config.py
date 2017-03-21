import sys
import logging as log
from relaax.common.python.config.base_config import BaseConfig


class CmdlConfig(BaseConfig):

    def __init__(self):
        super(CmdlConfig, self).__init__()

    def load_from_cmdl(self, parser):
        add = parser.add_argument
        add('-r', '--run', type=str, default='all', required=False,
            choices=('all', 'client', 'servers',
                     'rlx-server', 'parameter-server'),
            help='List of system components to run: client, servers, or all')
        add('-c', '--config', type=str, default=None, required=True,
            help='RELAAX config yaml file')
        add('-cl', '--client', type=str, default=None,
            help='Environment/Client to run')
        add('-cc', '--concurrent-clients', type=int, default=1,
            help='Number of environments/clients to run at the same time')
        add('-ll', '--log-level', type=str, default=None,
            choices=('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'),
            help='Logging level')

    def process_after_loaded(self):
        self.cmdl_log_level = self.get("log_level")

        if self.client is None:
            self.client = self.get("environment/client")

        if self.log_level is None:
            self.log_level = 'INFO'

    def setup_logger(self):
        log_level = getattr(log, 'INFO', None)

        if not isinstance(log_level, int):
            raise ValueError('Invalid log level: %s' % log_level)

        log.basicConfig(
            stream=sys.stdout,
            datefmt='%H:%M:%S',
            format='%(asctime)s %(name)s\t\t  | %(message)s',
            level=log_level)


options = CmdlConfig().load()
