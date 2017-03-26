import sys
import logging as log
from argparse import ArgumentParser

from config_yaml import ConfigYaml


class BaseConfig(ConfigYaml):

    def __init__(self):
        super(BaseConfig, self).__init__()

    def load_command_line(self):
        parser = ArgumentParser()
        self.load_from_cmdl(parser)
        self.merge_namespace(parser.parse_args())

    def load_from_cmdl(self, parser):
        add = parser.add_argument
        add('--log-level', type=str, default=None,
            choices=('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'),
            help='Logging level')
        add('--config', type=str, default=None, required=True,
            help='RELAAX config yaml file')
        add('--log-dir', type=str, default=None,
            help='Folder to store algorithm logs')
        add('--short-log-messages', type=bool, default=True,
            help='Log messages will skip long log prefix')

    def load_from_yaml(self):
        try:
            if self.config:
                self.load_from_file(self.config)
            self.process_after_loaded()
        except:
            log.critical("Can't load configuration file")
            raise

    def process_after_loaded(self):
        if self.log_level is None:
            self.log_level = 'DEBUG'

    def setup_logger(self):
        log_level = getattr(log, self.log_level.upper(), None)

        if not isinstance(log_level, int):
            raise ValueError('Invalid log level: %s' % self.log_level)

        if self.get('short_log_messages', False):
            format = '%(message)s'
        else:
            format = '[%(asctime)s]:[%(levelname)s]:[%(name)s]: %(message)s'

        log.basicConfig(
            stream=sys.stdout,
            datefmt='%H:%M:%S',
            format=format,
            level=log_level)

    def load(self):
        self.load_command_line()
        self.load_from_yaml()
        self.setup_logger()
        return self
