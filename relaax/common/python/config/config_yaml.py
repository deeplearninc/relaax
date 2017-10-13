import re
import ruamel.yaml
from argparse import Namespace

# ConfigYaml load yaml file into Namespace
# additional Namespace(s) could be merged in


class ConfigYaml(Namespace):
    def __init__(self):
        super(ConfigYaml, self).__init__()

    def load_from_file(self, filename):
        if not isinstance(filename, str) or len(filename) == 0:
            raise ValueError("please provide yaml file name")
        with open(filename, 'r') as f:
            self.load_to_namespace(
                self, ruamel.yaml.load(f, Loader=ruamel.yaml.Loader))

    def merge_namespace(self, namespace):
        for name in vars(namespace):
            setattr(self, name, getattr(namespace, name))

    # allow access to variable in namespace using name:
    # "var_name1/var_vame2"
    # if variable is not found, default will be returned
    def get(self, path, default=None):
        return ConfigYaml._get(self, path.split('/'), default)

    def load_to_namespace(self, obj, options):
        if isinstance(options, dict):
            for item in list(options.items()):
                self.load_to_namespace(obj, item)
        else:
            name = self._clean_name(options[0])
            if isinstance(options[1], dict):
                ns = Namespace()
                self.load_to_namespace(ns, options[1])
                setattr(obj, name, ns)
            else:
                setattr(obj, name, options[1])

    @staticmethod
    def _get(options, path, default):
        if len(path) == 0:
            return options
        if hasattr(options, path[0]):
            return ConfigYaml._get(
                getattr(options, path[0]), path[1:], default)
        return default

    @staticmethod
    def _clean_name(s):
        s = re.sub('\W|^(?=\d)', '_', s)
        s = re.sub('^[^A-Za-z]*', '', s)
        return s
