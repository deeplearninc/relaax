from __future__ import absolute_import

import tensorflow as tf

from relaax.common.algorithms import subgraph
from relaax.common.algorithms.lib import graph
from relaax.common.algorithms.lib import layer
from relaax.common.algorithms.lib import loss
from relaax.common.algorithms.lib import utils
from relaax.common.algorithms.lib import optimizer

from .dqn_config import config
from .lib.dqn_utils import Actor


class Network(subgraph.Subgraph):
    def build_graph(self):
        input = layer.Input(config.input)

        hidden = layer.GenericLayers(layer.Flatten(input),
                                     [dict(type=layer.Dense, size=size, activation=layer.Activation.Tanh)
                                      for size in config.hidden_sizes])

        weights = [input, hidden]

        if config.dueling_dqn:
            if config.hidden_sizes:
                v_input, a_input = tf.split(hidden.node,
                                            [config.hidden_sizes[-1] // 2, config.hidden_sizes[-1] // 2],
                                            axis=1)

                v_input = graph.TfNode(v_input)
                a_input = graph.TfNode(a_input)
            else:
                v_input, a_input = hidden, hidden

            v_output = layer.Dense(v_input, 1)
            a_output = layer.Dense(a_input, config.output.action_size)

            output = v_output.node + a_output.node - tf.reduce_mean(a_output.node, axis=1, keep_dims=True)
            output = graph.TfNode(output)

            weights.extend([v_output, a_output])
        else:
            output = layer.Dense(hidden, config.output.action_size)
            weights.append(output)

        self.ph_state = input.ph_state
        self.output = output
        self.weights = layer.Weights(*weights)


class GlobalServer(subgraph.Subgraph):
    def build_graph(self):
        sg_global_step = graph.GlobalStep()
        sg_network = Network()

        if config.optimizer == 'Adam':
            sg_optimizer = optimizer.AdamOptimizer(config.initial_learning_rate)
        elif config.optimizer == 'RMSProp':
            param = {}
            if hasattr(config, 'RMSProp'):
                if hasattr(config.RMSProp, "decay"):
                    param["decay"] = config.RMSProp.decay
                if hasattr(config.RMSProp, "epsilon"):
                    param["epsilon"] = config.RMSProp.epsilon

            sg_optimizer = optimizer.RMSPropOptimizer(config.initial_learning_rate, **param)
        else:
            raise NotImplementedError

        sg_gradients_apply = optimizer.Gradients(sg_network.weights, optimizer=sg_optimizer)

        sg_initialize = graph.Initialize()

        # Expose public API
        self.op_n_step = self.Op(sg_global_step.n)

        self.op_get_weights = self.Op(sg_network.weights)
        self.op_assign_weights = self.Op(sg_network.weights.assign,
                                         weights=sg_network.weights.ph_weights)

        self.op_apply_gradients = self.Op(sg_gradients_apply.apply, gradients=sg_gradients_apply.ph_gradients)

        self.op_initialize = self.Op(sg_initialize)


class AgentModel(subgraph.Subgraph):
    def build_graph(self):
        sg_network = Network()
        sg_target_network = Network()

        sg_get_action = Actor()

        sg_loss = loss.DQNLoss(sg_network.output, config)
        sg_gradients_calc = optimizer.Gradients(sg_network.weights, loss=sg_loss)

        sg_update_target_weights = graph.AssignWeights(sg_target_network.weights, sg_network.weights)

        # Expose public API
        self.op_assign_weights = self.Op(sg_network.weights.assign,
                                         weights=sg_network.weights.ph_weights)
        self.op_assign_target_weights = self.Op(sg_target_network.weights.assign,
                                                target_weights=sg_target_network.weights.ph_weights)

        self.op_get_q_value = self.Op(sg_network.output.node,
                                      state=sg_network.ph_state)
        self.op_get_q_target_value = self.Op(sg_target_network.output.node,
                                             next_state=sg_target_network.ph_state)

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

        self.op_update_target_weights = self.Op(sg_update_target_weights)

        self.op_initialize = self.Op(sg_initialize)


if __name__ == '__main__':
    utils.assemble_and_show_graphs(GlobalServer, AgentModel)
