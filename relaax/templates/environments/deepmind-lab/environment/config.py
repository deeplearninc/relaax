from relaax.common.python.config.base_config import BaseConfig


class DMLConfig(BaseConfig):

    def __init__(self):
        super(DMLConfig, self).__init__()

    def load_from_cmdl(self, parser):
        add = parser.add_argument
        add('--config', type=str, default=None, required=False,
            help='RELAAX config yaml file')
        add('--rlx-server-address', type=str, default=None, required=False,
            help='RELAAX RLX Server Address')
        add('--exploit', type='bool', default=False, metavar='True|False',
            help='Run client in exploit mode if set to True')
        add('--show-ui', type='bool', default=False, metavar='True|False',
            help='Client should show UI if set to True')
        add('--short-log-messages', type='bool', default=True, metavar='True|False',
            help='Log messages will skip long log prefix')

    def process_after_loaded(self):
        self.log_level = getattr(self, 'log_level', 'DEBUG')


options = DMLConfig().load()
