from builtins import object

import os.path

from . import class_loader


class AlgorithmLoader(object):

    @classmethod
    def load_agent(cls, algorithm_path, algorithm_name):
        return cls.load_algorithm_class(algorithm_path, algorithm_name, 'agent.Agent')

    @classmethod
    def load_parameter_server(cls, algorithm_path, algorithm_name):
        return cls.load_algorithm_class(algorithm_path, algorithm_name, 'parameter_server.ParameterServer')

    @classmethod
    def load_algorithm_class(cls, algorithm_path, algorithm_name, suffix):
        if algorithm_path is not None:
            _, algorithm_name = os.path.split(algorithm_path)
        if algorithm_path is None and '.' not in algorithm_name:
            algorithm_name = 'relaax.algorithms.%s' % algorithm_name
        return class_loader.ClassLoader.load(algorithm_path, '%s.%s' % (algorithm_name, suffix))
