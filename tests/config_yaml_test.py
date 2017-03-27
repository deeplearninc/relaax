from argparse import Namespace

from relaax.common.python.config.config_yaml import ConfigYaml


class TestConfigYaml:

    def setup_method(self, method):
        self.yaml = ConfigYaml()

    def test_load_from_file(self):
        self.yaml.load_from_file('tests/fixtures/fixture.yaml')
        assert self.yaml.relaax_rlx_server.bind == 'localhost:7001'

    def test_load_from_file_without_name(self):
        try:
            self.yaml.load_from_file(None)
            assert False
        except ValueError as e:
            assert str(e) == 'please provide yaml file name'
        except:
            assert False
        try:
            self.yaml.load_from_file('')
            assert False
        except ValueError as e:
            assert str(e) == 'please provide yaml file name'
        except:
            assert False

    def test_merge_namespace(self):
        ns = Namespace()
        ns.ns_attribute = 'ns value'
        self.yaml.yaml_attribyte = "yaml value"
        self.yaml.merge_namespace(ns)
        assert vars(self.yaml) == {
            'ns_attribute': 'ns value', 'yaml_attribyte': 'yaml value'}

    def test_get_with_path(self):
        value = 'value of attr2'
        self.yaml.attr1 = Namespace()
        self.yaml.attr1.attr2 = value
        assert self.yaml.get('attr1/attr2') == value

    def test_get_default_with_path(self):
        value = 'value of attr2'
        assert self.yaml.get('attr1/attr2', value) == value
