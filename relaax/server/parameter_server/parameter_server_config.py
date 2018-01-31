from relaax.common.python.config.base_config import BaseConfig


class ParameterServerConfig(BaseConfig):

    def __init__(self):
        super(ParameterServerConfig, self).__init__()

    def load_from_cmdl(self, parser):
        add = parser.add_argument
        add('--config', type=str, default=None, required=True,
            help='RELAAX config yaml file')
        add('--bind', type=str, default=None,
            help='Address of the parameter server in the format host:port')
        add('--metrics-server', type=str, default=None,
            help=('Address of the metrics server '
                  'address in the format host:port'))
        add('--log-level', type=str, default=None,
            choices=('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'),
            help='Logging level')
        add('--short-log-messages', type=bool, default=True,
            help='Log messages will skip long log prefix')

    def process_after_loaded(self):
        if self.log_level is None:
            self.log_level = self.get('relaax_parameter_server/log_level', 'DEBUG')

        if self.bind is None:
            self.bind = self.get_parameter_server()

        if self.metrics_server is None:
            self.metrics_server = self.get_metrics_server()

        self.algorithm_path = self.get('algorithm/path')
        self.algorithm_name = self.get('algorithm/name')

        self.define_missing('checkpoints_to_keep', None)
        self.define_missing('checkpoint_step_interval', None)
        self.define_missing('checkpoint_time_interval', None)
        self.define_missing('checkpoint_aws_s3', None)
        self.define_missing('checkpoint_dir', None)
        self.define_missing('best_checkpoints_to_keep', None)

        # Simple check of the bind address format
        self.bind = self.parse_address(self.bind, 'parameter server bind')
        self.metrics_server = self.parse_address(self.metrics_server, 'metrics server')

    def define_missing(self, key, value):
        if not hasattr(self.relaax_parameter_server, key):
            setattr(self.relaax_parameter_server, key, value)


options = ParameterServerConfig().load()
