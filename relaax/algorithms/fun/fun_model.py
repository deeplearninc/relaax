from __future__ import absolute_import
import tensorflow as tf

from relaax.common.algorithms import subgraph
from relaax.common.algorithms.lib import graph
from relaax.common.algorithms.lib import layer
from relaax.common.algorithms.lib import utils

from . import fun_config
from .lib import da3c_graph


class PerceptionNetwork(subgraph.Subgraph):
    def build_graph(self):
        input = layer.Input(fun_config.config.input)

        self.perception =\
            layer.Dense(layer.Flatten(input), fun_config.config.d,  # d=256
                        activation=layer.Activation.Relu)

        self.weights = layer.Weigths(input, self.perception)


class ManagerNetwork(subgraph.Subgraph):
    def build_graph(self):
        self.ph_perception =\
            tf.placeholder(tf.float32, shape=[None, fun_config.config.d],
                           name="ph_perception")

        self.Mspace = \
            layer.Dense(self.ph_perception, fun_config.config.d,  # d=256
                        activation=layer.Activation.Relu)


if __name__ == '__main__':
    utils.assemble_and_show_graphs()
