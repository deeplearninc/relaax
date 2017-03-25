import sys
import logging as log
from argparse import Namespace

from fixtures.mock_cmdl import cmdl
from fixtures.mock_utils import MockUtils
from relaax.common.python.config.base_config import BaseConfig


class TestBaseConfig:

    @classmethod
    def teardown_class(cls):
        cmdl.restore()

    def setup_method(self, method):
        self.config = BaseConfig()

    def test_load_from_cmdl(self):
        cmdl.set_args([
            'base_config_test.py',
            '--config', 'tests/fixtures/fixture.yaml'])
        self.config.load_command_line()
        assert vars(self.config) == {
            'log_level': None,
            'config': 'tests/fixtures/fixture.yaml',
            'log_dir': None, 'short_log_messages': True}

    def test_load_from_yaml(self):
        self.config.config = 'tests/fixtures/short.yaml'
        self.config.log_level = None
        self.config.load_from_yaml()
        assert vars(self.config) == {
            'config': 'tests/fixtures/short.yaml',
            'environment': Namespace(client='client/sample_exchange.py'),
            'log_level': 'DEBUG',
            'relaax_rlx_server': Namespace(bind='localhost:7001', log_level='DEBUG')}

    def test_load_from_yaml_without_config_file(self):
        try:
            self.config.config = None
            self.config.load_from_yaml()
            assert False
        except Exception as e:
            assert str(e) == 'please provide yaml file name'

    def test_process_after_loaded(self):
        self.config.log_level = None
        self.config.process_after_loaded()
        assert self.config.log_level == 'DEBUG'
        self.config.log_level = 'WARNING'
        self.config.process_after_loaded()
        assert self.config.log_level == 'WARNING'

    def test_setup_logger(self, monkeypatch):
        called_with = MockUtils.called_with(log, 'basicConfig', monkeypatch)
        self.config.log_level = 'WARNING'
        self.config.setup_logger()
        assert called_with.kwargs == {
            'datefmt': '%H:%M:%S',
            'format': '[%(asctime)s]:[%(levelname)s]:[%(name)s]: %(message)s',
            'level': 30, 'stream': sys.stdout}

    def test_setup_logger_short_message(self, monkeypatch):
        called_with = MockUtils.called_with(log, 'basicConfig', monkeypatch)
        self.config.log_level = 'WARNING'
        self.config.short_log_messages = True
        self.config.setup_logger()
        assert called_with.kwargs == {
            'datefmt': '%H:%M:%S',
            'format': '%(message)s',
            'level': 30, 'stream': sys.stdout}

    def test_setup_logger_with_wrong_log_level(self, monkeypatch):
        try:
            MockUtils.called_with(log, 'basicConfig', monkeypatch)
            self.config.log_level = 'WRONG'
            self.config.setup_logger()
            assert False
        except Exception as e:
            assert str(e) == 'Invalid log level: WRONG'

    def test_load(self, monkeypatch):
        called_load_command_line = MockUtils.called_once(self.config, 'load_command_line', monkeypatch)
        called_load_from_yaml = MockUtils.called_once(self.config, 'load_from_yaml', monkeypatch)
        called_setup_logger = MockUtils.called_once(self.config, 'setup_logger', monkeypatch)
        self.config.load()
        assert called_load_command_line[0] == 1
        assert called_load_from_yaml[0] == 1
        assert called_setup_logger[0] == 1
