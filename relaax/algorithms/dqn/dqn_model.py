from __future__ import absolute_import

import tensorflow as tf

from relaax.common.algorithms import subgraph
from relaax.common.algorithms.lib import graph
from relaax.common.algorithms.lib import layer
from relaax.common.algorithms.lib import loss
from relaax.common.algorithms.lib import utils
from relaax.common.algorithms.lib import optimizer
from relaax.common.algorithms.lib import lr_schedule

from .dqn_config import config
from .lib.dqn_utils import Actor


class Network(subgraph.Subgraph):
    def build_graph(self):
        input = layer.ConfiguredInput(config.input)

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

        sg_get_weights_flatten = graph.GetVariablesFlatten(sg_network.weights)
        sg_set_weights_flatten = graph.SetVariablesFlatten(sg_network.weights)

        if config.use_linear_schedule:
            sg_learning_rate = lr_schedule.Linear(sg_global_step, config)
        else:
            sg_learning_rate = config.initial_learning_rate

        if config.optimizer == 'Adam':
            sg_optimizer = optimizer.AdamOptimizer(sg_learning_rate)
        elif config.optimizer == 'RMSProp':
            sg_optimizer = optimizer.RMSPropOptimizer(learning_rate=sg_learning_rate,
                                                      decay=config.RMSProp.decay,
                                                      epsilon=config.RMSProp.epsilon)
        else:
            assert False, 'There 2 valid options for optimizers: Adam | RMSProp'

        sg_gradients_apply = optimizer.Gradients(sg_network.weights, optimizer=sg_optimizer)

        sg_average_reward = graph.LinearMovingAverage(config.avg_in_num_batches)
        sg_initialize = graph.Initialize()

        # Expose public API
        self.op_n_step = self.Op(sg_global_step.n)
        self.op_score = self.Op(sg_average_reward.average)

        self.op_get_weights_signed = self.Ops(sg_network.weights, sg_global_step.n)
        self.op_assign_weights = self.Op(sg_network.weights.assign,
                                         weights=sg_network.weights.ph_weights)

        self.op_apply_gradients = self.Ops(sg_gradients_apply.apply, sg_global_step.increment,
                                           gradients=sg_gradients_apply.ph_gradients,
                                           increment=sg_global_step.ph_increment)
        self.op_add_rewards_to_model_score_routine = self.Ops(sg_average_reward.add,
                                                              reward_sum=sg_average_reward.ph_sum,
                                                              reward_weight=sg_average_reward.ph_count)

        self.op_get_weights_flatten = self.Op(sg_get_weights_flatten)
        self.op_set_weights_flatten = self.Op(sg_set_weights_flatten, value=sg_set_weights_flatten.ph_value)

        # Gradient combining routines
        self.op_submit_gradients = self.Call(graph.get_gradients_apply_routine(config))

        self.op_initialize = self.Op(sg_initialize)


class AgentModel(subgraph.Subgraph):
    def build_graph(self):
        sg_network = Network()
        sg_target_network = Network()

        sg_get_action = Actor()

        sg_loss = loss.DQNLoss(sg_network.output, config)
        sg_gradients_calc = optimizer.Gradients(sg_network.weights, loss=sg_loss)

        sg_update_target_weights = graph.AssignWeights(sg_target_network.weights, sg_network.weights).op

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
