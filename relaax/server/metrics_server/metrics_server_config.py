from relaax.common.python.config.base_config import BaseConfig


class MetricsServerConfig(BaseConfig):

    def __init__(self):
        super(MetricsServerConfig, self).__init__()

    def load_from_cmdl(self, parser):
        add = parser.add_argument
        add('--config', type=str, default=None, required=True,
            help='RELAAX config yaml file')
        add('--bind', type=str, default=None,
            help='Address of the parameter server in the format host:port')
        add('--log-level', type=str, default=None,
            choices=('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'),
            help='Logging level')
        add('--short-log-messages', type=bool, default=True,
            help='Log messages will skip long log prefix')

    def process_after_loaded(self):
        if self.log_level is None:
            self.log_level = self.get('relaax_metrics_server/log_level', 'DEBUG')

        if self.bind is None:
            self.bind = self.get_metrics_server()

        # Simple check of the bind address format
        self.bind = self.parse_address(self.bind, 'metrics server bind')


options = MetricsServerConfig().load()
