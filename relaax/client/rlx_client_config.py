from relaax.common.python.config.base_config import BaseConfig


class RlxClientConfig(BaseConfig):

    def __init__(self):
        super(RlxClientConfig, self).__init__()

    def load_from_cmdl(self, parser):
        add = parser.add_argument
        add('--config', type=str, default=None, required=False,
            help='RELAAX config yaml file')
        add('--log-level', type=str, default=None,
            choices=('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'),
            help='Logging level')
        add('--short-log-messages', type=bool, default=True,
            help='Log messages will skip long log prefix')

    def process_after_loaded(self):
        if self.log_level is None:
            self.log_level = 'DEBUG'


options = RlxClientConfig().load()
