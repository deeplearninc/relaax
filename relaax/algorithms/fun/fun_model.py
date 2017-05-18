from __future__ import absolute_import
import tensorflow as tf

from relaax.common.algorithms import subgraph
from relaax.common.algorithms.lib import graph
from relaax.common.algorithms.lib import layer
from relaax.common.algorithms.lib import utils

from .fun_config import config as cfg
from .lstm import DilatedLSTMCell, CustomBasicLSTMCell


class PerceptionNetwork(subgraph.Subgraph):
    def build_graph(self):
        input = layer.Input(cfg.input)

        self.perception =\
            layer.Dense(layer.Flatten(input), cfg.d,  # d=256
                        activation=layer.Activation.Relu)

        self.weights = layer.Weigths(input, self.perception)


class ManagerNetwork(subgraph.Subgraph):
    def build_graph(self):
        self.ph_perception =\
            tf.placeholder(tf.float32, shape=[None, cfg.d],
                           name="ph_perception")

        self.Mspace = \
            layer.Dense(self.ph_perception, cfg.d,  # d=256
                        activation=layer.Activation.Relu)

        self.lstm = DilatedLSTMCell(cfg.d, num_cores=cfg.d)

        self.step_size = tf.placeholder(tf.float32, [1], name="manager_step_size")
        self.initial_lstm_state = tf.placeholder(tf.float32, [1, self.lstm.state_size],
                                                 name="manager_lstm_state")

        scope = "manager_" + self.ph_perception.name[12:]

if __name__ == '__main__':
    utils.assemble_and_show_graphs()
