from __future__ import absolute_import

import tensorflow as tf

from relaax.common.algorithms import subgraph
from relaax.common.algorithms.lib import graph
from relaax.common.algorithms.lib import layer
from relaax.common.algorithms.lib import loss
from relaax.common.algorithms.lib import utils
from . import dqn_config as cfg

from .lib.dqn_utils import GetAction

class Network(subgraph.Subgraph):
    def build_graph(self):
        input = layer.Input(cfg.config.input)

        sizes = cfg.config.hidden_sizes

        hidden = layer.GenericLayers(layer.Flatten(input),
                                     [dict(type=layer.Dense, size=size,
                                           activation=layer.Activation.Tanh) for size in sizes])

        output = layer.Dense(hidden, cfg.config.output.action_size)

        layers = [input, hidden, output]

        self.ph_state = input.ph_state
        self.output = output
        self.weights = layer.Weights(*layers)


class GlobalServer(subgraph.Subgraph):
    def build_graph(self):
        sg_global_step = graph.GlobalStep()
        sg_network = Network()
        sg_target_network = Network()
        sg_weights = sg_network.weights
        sg_target_weights = sg_target_network.weights

        sg_update_target_weights = graph.TfNode([tf.assign(variable, value) for variable, value in utils.Utils.izip(sg_target_weights.node, sg_weights.node)])

        if cfg.config.optimizer == 'Adam':
            sg_optimizer = graph.AdamOptimizer(cfg.config.initial_learning_rate)
        else:
            raise NotImplementedError

        sg_gradients = layer.Gradients(sg_weights, optimizer=sg_optimizer)

        sg_initialize = graph.Initialize()

        # Expose public API
        self.op_n_step = self.Op(sg_global_step.n)

        self.op_get_weights = self.Op(sg_weights)
        self.op_get_target_weights = self.Op(sg_target_weights)

        self.op_update_target_weights = self.Op(sg_update_target_weights)
        self.op_apply_gradients = self.Ops(sg_gradients.apply, sg_global_step.increment,
                                           gradients=sg_gradients.ph_gradients,
                                           increment=sg_global_step.ph_increment)
        self.op_initialize = self.Op(sg_initialize)


# Policy run by Agent(s)
class AgentModel(subgraph.Subgraph):
    def build_graph(self):
        # Build graph
        sg_network = Network()
        sg_target_network = Network()

        sg_get_action = GetAction()

        sg_loss = loss.DQNLoss(sg_network.output, cfg.config.output, cfg.config.double_dqn, cfg.config.rewards_gamma)
        sg_gradients = layer.Gradients(sg_network.weights, loss=sg_loss)

        # Expose public API
        self.op_assign_weights = self.Op(sg_network.weights.assign, weights=sg_network.weights.ph_weights)
        self.op_assign_target_weights = self.Op(sg_target_network.weights.assign, target_weights=sg_target_network.weights.ph_weights)

        self.op_get_q_value = self.Op(sg_network.output.node, state=sg_network.ph_state)
        self.op_get_q_target_value = self.Op(sg_target_network.output.node, next_state=sg_target_network.ph_state)
        self.op_get_action = self.Op(sg_get_action,
                                     global_step=sg_get_action.ph_global_step,
                                     q_value=sg_get_action.ph_q_value)

        sg_initialize = graph.Initialize()

        feeds = dict(state=sg_network.ph_state,
                     reward=sg_loss.ph_reward,
                     action=sg_loss.ph_action,
                     terminal=sg_loss.ph_terminal,
                     q_next_target=sg_loss.ph_q_next_target)

        if cfg.config.double_dqn:
            feeds["q_next"] = sg_loss.ph_q_next

        self.op_compute_gradients = self.Op(sg_gradients.calculate, **feeds)

        self.op_initialize = self.Op(sg_initialize)


if __name__ == '__main__':
    utils.assemble_and_show_graphs(GlobalServer, AgentModel)
