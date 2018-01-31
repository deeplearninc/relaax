from __future__ import absolute_import
import argparse
import logging
import os
import re
import sys
import relaax

from .config_yaml import ConfigYaml


logger = logging.getLogger(__name__)


class RelaaxHelpFormatter(argparse.HelpFormatter):

    def _format_action_invocation(self, action):
        if not action.option_strings:
            metavar, = self._metavar_formatter(action, action.dest)(1)
            return metavar
        else:
            parts = []
            if action.nargs == 0:
                parts.extend(action.option_strings)
            else:
                default = action.dest.upper()
                args_string = self._format_args(action, default)
                for option_string in action.option_strings:
                    parts.append('%s' % option_string)
                parts_string = ', '.join(parts)
                parts_string = parts_string.ljust(
                    self._max_help_position - self._current_indent)
                return "%s[%s]" % (parts_string, args_string)
            return ', '.join(parts)


class BaseConfig(ConfigYaml):

    def __init__(self):
        super(BaseConfig, self).__init__()

    def load_command_line(self):
        parser = argparse.ArgumentParser(formatter_class=RelaaxHelpFormatter)
        # add type keyword to registries
        parser.register('type', 'bool', BaseConfig.str2bool)
        self.load_from_cmdl(parser)
        self.merge_namespace(parser.parse_args())

    def load_from_cmdl(self, parser):
        add = parser.add_argument
        add('--log-level', type=str, default=None,
            choices=('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'),
            help='Logging level.')
        add('--config', type=str, default=None, required=True,
            help='Relaax configuration yaml file.')
        add('--log-dir', type=str, default=None,
            help='Folder to store log files.')
        add('--short-log-messages', type='bool', default=True, metavar='True|False',
            help='Log messages will skip long logging prefix.')

    def load_from_yaml(self):
        try:
            if self.config:
                self.load_from_file(self.config)
            self.process_after_loaded()
        except Exception:
            logger.critical("Can't load configuration file")
            raise

    def process_after_loaded(self):
        if self.log_level is None:
            self.log_level = 'DEBUG'

    def setup_logger(self):
        log_level = getattr(logging, self.log_level.upper(), None)

        if not isinstance(log_level, int):
            raise ValueError('Invalid logging level: %s' % self.log_level)

        if self.get('short_log_messages', False):
            format = '%(message)s'
        else:
            format = '[%(asctime)s]:[%(levelname)s]:[%(name)s]: %(message)s'

        logging.basicConfig(stream=sys.stdout, datefmt='%H:%M:%S', format=format, level=log_level)

    def get_binding(self, section, def_value):
        bind = self.get('%s/bind' % section, 'localhost:7002')
        bind_by_env_var = self.get('%s/bind_by_env_var' % section, None)
        if bind_by_env_var is not None and bind_by_env_var in os.environ:
            bind = os.environ[bind_by_env_var]
        return bind

    def get_parameter_server(self):
        return self.get_binding('relaax_parameter_server', 'localhost:7000')

    def get_rlx_server(self):
        return self.get_binding('relaax_rlx_server', 'localhost:7001')

    def get_metrics_server(self):
        return self.get_binding('relaax_metrics_server', 'localhost:7002')

    def parse_address(self, address, address_name):
        m = re.match('^\s*([^\s:]*)\s*:\s*(\d+)\s*$', address)
        if m is None:
            logger.error('Please specify %s address in host:port format.', address_name)
            exit(1)
        return m.group(1), int(m.group(2))

    def load(self):
        self.load_command_line()
        self.load_from_yaml()
        self.setup_logger()
        # check version of YAML only when YAML was loaded
        if getattr(self, 'config', None) is not None:
            assert relaax.__version__ == self.get('version'),\
                'You have to provide appropriate RELAAX version X.X.X in yaml'
        return self

    @staticmethod
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        if v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentError('Boolean value expected.')
