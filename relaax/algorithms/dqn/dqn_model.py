from __future__ import absolute_import

import tensorflow as tf

from relaax.common.algorithms import subgraph
from relaax.common.algorithms.lib import graph
from relaax.common.algorithms.lib import layer
from relaax.common.algorithms.lib import loss
from relaax.common.algorithms.lib import utils

from . import dqn_config as cfg
from .lib.dqn_utils import Actor


class Network(subgraph.Subgraph):
    def build_graph(self):
        input = layer.Input(cfg.config.input)

        hidden = layer.GenericLayers(layer.Flatten(input),
                                     [dict(type=layer.Dense, size=size, activation=layer.Activation.Tanh) for size in cfg.config.hidden_sizes])

        output = layer.Dense(hidden, cfg.config.output.action_size)

        self.ph_state = input.ph_state
        self.output = output
        self.weights = layer.Weights(input, hidden, output)


class GlobalServer(subgraph.Subgraph):
    def build_graph(self):
        sg_global_step = graph.GlobalStep()
        sg_target_network = Network()

        sg_initialize = graph.Initialize()

        # Expose public API
        self.op_n_step = self.Op(sg_global_step.n)

        self.op_get_target_weights = self.Op(sg_target_network.weights)
        self.op_assign_target_weights = self.Op(sg_target_network.weights.assign, target_weights=sg_target_network.weights.ph_weights)

        self.op_initialize = self.Op(sg_initialize)


class AgentModel(subgraph.Subgraph):
    def build_graph(self):
        sg_network = Network()
        sg_target_network = Network()

        sg_get_action = Actor()

        if cfg.config.optimizer == 'Adam':
            sg_optimizer = graph.AdamOptimizer(cfg.config.initial_learning_rate)
        else:
            raise NotImplementedError

        sg_loss = loss.DQNLoss(sg_network.output, cfg.config)
        sg_gradients_calc = layer.Gradients(sg_network.weights, loss=sg_loss)
        sg_gradients_apply = layer.Gradients(sg_network.weights, optimizer=sg_optimizer)

        sg_update_target_weights = graph.TfNode([tf.assign(variable, value) for variable, value in utils.Utils.izip(sg_target_network.weights.node, sg_network.weights.node)])

        # Expose public API
        self.op_assign_weights = self.Op(sg_network.weights.assign, weights=sg_network.weights.ph_weights)
        self.op_assign_target_weights = self.Op(sg_target_network.weights.assign, target_weights=sg_target_network.weights.ph_weights)

        self.op_get_weights = self.Op(sg_network.weights)

        self.op_get_q_value = self.Op(sg_network.output.node, state=sg_network.ph_state)
        self.op_get_q_target_value = self.Op(sg_target_network.output.node, next_state=sg_target_network.ph_state)

        self.op_get_action = self.Op(sg_get_action,
                                     local_step=sg_get_action.ph_local_step,
                                     q_value=sg_get_action.ph_q_value)

        sg_initialize = graph.Initialize()

        feeds = dict(state=sg_network.ph_state,
                     reward=sg_loss.ph_reward,
                     action=sg_loss.ph_action,
                     terminal=sg_loss.ph_terminal,
                     q_next_target=sg_loss.ph_q_next_target,
                     q_next=sg_loss.ph_q_next)

        self.op_compute_gradients = self.Op(sg_gradients_calc.calculate, **feeds)
        self.op_apply_gradients = self.Op(sg_gradients_apply.apply, gradients=sg_gradients_apply.ph_gradients)

        self.op_update_target_weights = self.Op(sg_update_target_weights)

        self.op_initialize = self.Op(sg_initialize)


if __name__ == '__main__':
    utils.assemble_and_show_graphs(GlobalServer, AgentModel)
