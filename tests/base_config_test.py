from __future__ import absolute_import
from builtins import str
from builtins import object
from argparse import Namespace
import errno
import logging as log
import sys

from .fixtures.mock_cmdl import cmdl
from .fixtures.mock_utils import MockUtils
from relaax.common.python.config.base_config import BaseConfig


class TestBaseConfig(object):

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

    def test_load_from_yaml_with_wrog_config_file_name(self):
        try:
            self.config.config = "wrong file name"
            self.config.load_from_yaml()
            assert False
        except Exception as e:
            assert str(e) == '[Errno %d] No such file or directory: \'wrong file name\'' % errno.ENOENT

    def test_load_from_yaml_with_no_config_file_name(self):
        try:
            self.config.config = None
            self.config.log_level = None
            self.config.load_from_yaml()
            assert True
        except Exception:
            assert False

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
            assert str(e) == 'Invalid logging level: WRONG'

    def test_load(self, monkeypatch):
        called_load_command_line = MockUtils.count_calls(self.config, 'load_command_line', monkeypatch)
        called_load_from_yaml = MockUtils.count_calls(self.config, 'load_from_yaml', monkeypatch)
        called_setup_logger = MockUtils.count_calls(self.config, 'setup_logger', monkeypatch)
        self.config.load()
        assert called_load_command_line.times == 1
        assert called_load_from_yaml.times == 1
        assert called_setup_logger.times == 1
