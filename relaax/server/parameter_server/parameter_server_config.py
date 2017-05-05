import logging as log

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
        add('--log-level', type=str, default=None,
            choices=('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'),
            help='Logging level')
        add('--short-log-messages', type=bool, default=True,
            help='Log messages will skip long log prefix')

    def process_after_loaded(self):
        if self.log_level is None:
            self.log_level = self.get(
                'relaax_parameter_server/log_level', 'DEBUG')

        if self.bind is None:
            self.bind = self.get(
                'relaax_parameter_server/bind', 'localhost:7000')

        self.algorithm_path = self.get('algorithm/path')

        self.define_missing('checkpoints_to_keep', None)
        self.define_missing('checkpoint_step_interval', None)
        self.define_missing('checkpoint_time_interval', None)
        self.define_missing('checkpoint_aws_s3', None)
        self.define_missing('checkpoint_dir', None)

        # Simple check of the bind address format

        self.bind = map(lambda x: x.strip(), self.bind.split(':'))
        if len(self.bind) != 2:
            log.error(
                ('Please specify parameter server'
                 'bind address in host:port format'))
            exit()
        self.bind[1] = int(self.bind[1])
        self.bind = tuple(self.bind)

    def define_missing(self, key, value):
        if not hasattr(self.relaax_parameter_server, key):
            setattr(self.relaax_parameter_server, key, value)


options = ParameterServerConfig().load()
