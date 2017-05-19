from __future__ import absolute_import
import tensorflow as tf
import numpy as np

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

        self.weights = layer.Weights(input, self.perception)


class ManagerNetwork(subgraph.Subgraph):
    def build_graph(self):
        self.ph_perception =\
            graph.Placeholder(np.float32, shape=(None, cfg.d), name="ph_perception")
        # tf.placeholder(tf.float32, shape=[None, cfg.d], name="ph_perception")

        self.Mspace =\
            layer.Dense(self.ph_perception, cfg.d,  # d=256
                        activation=layer.Activation.Relu)
        Mspace_expanded = tf.expand_dims(self.Mspace, 0)

        self.lstm = DilatedLSTMCell(cfg.d, num_cores=cfg.d)
        # needs wrap as layer to retrieve weights

        self.step_size =\
            graph.Placeholder(np.float32, shape=(1,), name="ph_m_step_size")
        # tf.placeholder(tf.float32, [1], name="ph_m_step_size")
        self.initial_lstm_state = \
            graph.Placeholder(np.float32, shape=(1, self.lstm.state_size), name="ph_m_lstm_state")
        # tf.placeholder(tf.float32, [1, self.lstm.state_size], name="ph_m_lstm_state")

        lstm_outputs, self.lstm_state = tf.nn.dynamic_rnn(self.lstm,
                                                          Mspace_expanded,
                                                          initial_state=self.initial_lstm_state,
                                                          sequence_length=self.step_size,
                                                          time_major=False)
        sg_lstm_outputs = graph.TfNode(lstm_outputs)

        self.goal = tf.nn.l2_normalize(graph.Flatten(sg_lstm_outputs), dim=1)

        critic = layer.Dense(sg_lstm_outputs, 1)
        self.critic = layer.Flatten(critic)

        self.weights = layer.Weights(self.Mspace,
                                     graph.TfNode((self.lstm.matrix, self.lstm.bias)),
                                     critic)

        self.lstm_state_out = np.zeros([1, self.lstm.state_size])

if __name__ == '__main__':
    utils.assemble_and_show_graphs()
