import sys

from fixtures.mock_cmdl import cmdl

from relaax.common.python.config.loaded_config import AlgoritmConfig


class TestAlgoritmConfig:
    rlx = 'relaax.server.rlx_server.rlx_config'
    ps = 'relaax.server.parameter_server.parameter_server_config'

    @classmethod
    def teardown_class(cls):
        cmdl.restore()

    def setup_method(self, method):
        if self.rlx in sys.modules:
            del sys.modules[self.rlx]
        if self.ps in sys.modules:
            del sys.modules[self.ps]
        cmdl.set_args([
            'loaded_config_test.py',
            '--config', 'tests/fixtures/fixture.yaml'])

    def test_dev_config(self):
        options = AlgoritmConfig.select_config()
        assert type(options).__name__ == 'DevConfig'

    def test_rlx_config(self):
        from relaax.server.rlx_server.rlx_config import options as rlx_options
        options = AlgoritmConfig.select_config()
        assert type(options).__name__ == 'RLXConfig'
        assert rlx_options == options

    def test_ps_config(self):
        from relaax.server.parameter_server.parameter_server_config import options as ps_options
        options = AlgoritmConfig.select_config()
        assert type(options).__name__ == 'ParameterServerConfig'
        assert options == ps_options
