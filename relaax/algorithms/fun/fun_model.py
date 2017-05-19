from __future__ import absolute_import
import tensorflow as tf
import numpy as np

from relaax.common.algorithms import subgraph
from relaax.common.algorithms.lib import graph
from relaax.common.algorithms.lib import layer
from relaax.common.algorithms.lib import utils

from .lib import fun_graph
from .fun_config import config as cfg
from .lstm import DilatedLSTMCell, CustomBasicLSTMCell


class _PerceptionNetwork(subgraph.Subgraph):
    def build_graph(self):
        input = layer.Input(cfg.input)

        self.perception =\
            layer.Dense(layer.Flatten(input), cfg.d,  # d=256
                        activation=layer.Activation.Relu)

        self.weights = layer.Weights(input, self.perception)


class _ManagerNetwork(subgraph.Subgraph):
    def build_graph(self):
        self.ph_perception =\
            graph.Placeholder(np.float32, shape=(None, cfg.d), name="ph_perception")
        # tf.placeholder(tf.float32, shape=[None, cfg.d], name="ph_perception")

        self.Mspace =\
            layer.Dense(self.ph_perception.node, cfg.d,  # d=256
                        activation=layer.Activation.Relu)
        Mspace_expanded = tf.expand_dims(self.Mspace, 0)

        self.lstm = DilatedLSTMCell(cfg.d, num_cores=cfg.d)
        # needs wrap as layer to retrieve weights

        self.ph_step_size =\
            graph.Placeholder(np.float32, shape=(1,), name="ph_m_step_size")
        # tf.placeholder(tf.float32, [1], name="ph_m_step_size")
        self.ph_initial_lstm_state =\
            graph.Placeholder(np.float32, shape=(1, self.lstm.state_size), name="ph_m_lstm_state")
        # tf.placeholder(tf.float32, [1, self.lstm.state_size], name="ph_m_lstm_state")

        lstm_outputs, self.lstm_state = tf.nn.dynamic_rnn(self.lstm,
                                                          Mspace_expanded,
                                                          initial_state=self.ph_initial_lstm_state,
                                                          sequence_length=self.ph_step_size,
                                                          time_major=False)
        sg_lstm_outputs = graph.TfNode(lstm_outputs)

        self.goal = tf.nn.l2_normalize(graph.Flatten(sg_lstm_outputs.node), dim=1)

        critic = layer.Dense(sg_lstm_outputs.node, 1)
        self.value = layer.Flatten(critic)

        self.weights = layer.Weights(self.Mspace,
                                     graph.TfNode((self.lstm.matrix, self.lstm.bias)),
                                     critic)

        self.lstm_state_out = np.zeros([1, self.lstm.state_size])


class GlobalManagerNetwork(subgraph.Subgraph):
    def build_graph(self):
        sg_weights = _ManagerNetwork().weights

        sg_global_step = graph.GlobalStep()
        # self.learning_rate_input = graph.Placeholder(np.float32, shape=(1,), name="manager_lr")
        # tf.placeholder(tf.float32, [], name="manager_lr")
        sg_learning_rate = fun_graph.LearningRate(sg_global_step)

        sg_optimizer = graph.RMSPropOptimizer(
            learning_rate=sg_learning_rate,
            decay=cfg.RMSProp.decay,
            momentum=0.0,
            epsilon=cfg.RMSProp.epsilon
        )

        sg_gradients = layer.Gradients(sg_weights, optimizer=sg_optimizer)
        sg_initialize = graph.Initialize()

        # Expose public API
        self.op_n_step = self.Op(sg_global_step.n)
        self.op_get_weights = self.Op(sg_weights)
        self.op_apply_gradients = self.Ops(sg_gradients.apply,
                                           sg_global_step.increment,
                                           gradients=sg_gradients.ph_gradients,
                                           increment=sg_global_step.ph_increment)
        self.op_initialize = self.Op(sg_initialize)


class LocalManagerNetwork(subgraph.Subgraph):
    def build_graph(self):
        self.sg_network = _ManagerNetwork()

        sg_loss = fun_graph.ManagerLoss(self.sg_network.goal, self.sg_network.value)
        sg_gradients = layer.Gradients(self.sg_network.weights, loss=sg_loss)

        # Expose public API
        self.op_assign_weights = self.Op(self.sg_network.weights.assign,
                                         weights=self.sg_network.weights.ph_weights)
        self.op_compute_gradients =\
            self.Op(sg_gradients.calculate,
                    ph_perception=self.sg_network.ph_perception,
                    ph_stc_diff_st=sg_loss.ph_stc_diff_st,
                    ph_discounted_reward=sg_loss.ph_discounted_reward,
                    ph_initial_lstm_state=self.sg_network.ph_initial_lstm_state,
                    ph_step_size=self.sg_network.ph_step_size)

        # Need MORE with lstm's freezes
        self.op_get_goal_value_st = self.Ops(
            self.sg_network.goal, self.sg_network.value, self.sg_network.Mspace,
            ph_perception=self.sg_network.ph_perception,
            ph_initial_lstm_state=self.sg_network.ph_initial_lstm_state,
            ph_step_size=self.sg_network.ph_step_size)
        self.op_get_st = self.Op(
            self.sg_network.Mspace,
            ph_perception=self.sg_network.ph_perception)
        # self.op_get_goal_st
        # self.op_get_value

    def reset_state(self):
        self.sg_network.lstm_state_out = np.zeros([1, self.sg_network.lstm.state_size])


if __name__ == '__main__':
    utils.assemble_and_show_graphs()
