from __future__ import absolute_import

import logging

from relaax.common.algorithms import subgraph
from relaax.common.algorithms.lib import graph
from relaax.common.algorithms.lib import layer
from relaax.common.algorithms.lib import optimizer
from relaax.common.algorithms.lib import loss
from relaax.common.algorithms.lib import utils

from . import pg_config

logger = logging.getLogger(__name__)


class Network(subgraph.Subgraph):
    def build_graph(self):
        input = layer.ConfiguredInput(pg_config.config.input)

        dense = layer.GenericLayers(layer.Flatten(input),
                                    [dict(type=layer.Dense, size=size, activation=layer.Activation.Relu)
                                    for size in pg_config.config.hidden_sizes])

        actor = layer.Dense(dense, pg_config.config.output.action_size,
                            activation=layer.Activation.Softmax)

        self.state = input.ph_state
        self.weights = layer.Weights(input, dense, actor)
        return actor.node


# Weights of the policy are shared across
# all agents and stored on the parameter server
class SharedParameters(subgraph.Subgraph):
    def build_graph(self):
        # Build graph
        sg_global_step = graph.GlobalStep()
        sg_weights = Network().weights

        # Weights get/set for updating the policy
        sg_get_weights_flatten = graph.GetVariablesFlatten(sg_weights)
        sg_set_weights_flatten = graph.SetVariablesFlatten(sg_weights)

        sg_optimizer = optimizer.AdamOptimizer(pg_config.config.learning_rate)
        sg_gradients = optimizer.Gradients(sg_weights, optimizer=sg_optimizer)

        sg_average_reward = graph.LinearMovingAverage(pg_config.config.avg_in_num_batches)
        sg_initialize = graph.Initialize()

        # Expose public API
        self.op_n_step = self.Op(sg_global_step.n)
        self.op_score = self.Op(sg_average_reward.average)

        self.op_get_weights_signed = self.Ops(sg_weights, sg_global_step.n)
        self.op_apply_gradients = self.Ops(sg_gradients.apply, sg_global_step.increment,
                                           gradients=sg_gradients.ph_gradients,
                                           increment=sg_global_step.ph_increment)
        self.op_add_rewards_to_model_score_routine = self.Ops(sg_average_reward.add,
                                                              reward_sum=sg_average_reward.ph_sum,
                                                              reward_weight=sg_average_reward.ph_count)
        # Ops get/set for updating the policy
        self.op_get_weights_flatten = self.Op(sg_get_weights_flatten)
        self.op_set_weights_flatten = self.Op(sg_set_weights_flatten, value=sg_set_weights_flatten.ph_value)

        # Gradient combining routines
        self.op_submit_gradients = self.Call(graph.get_gradients_apply_routine(pg_config.config))

        self.op_initialize = self.Op(sg_initialize)


# Policy run by Agent(s)
class PolicyModel(subgraph.Subgraph):
    def build_graph(self):
        # Build graph
        sg_network = Network()

        sg_loss = loss.PGLoss(action_size=pg_config.config.output.action_size,
                              network=sg_network)
        sg_gradients = optimizer.Gradients(sg_network.weights, loss=sg_loss)

        # Expose public API
        self.op_assign_weights = self.Op(sg_network.weights.assign,
                                         weights=sg_network.weights.ph_weights)
        self.op_get_action = self.Op(sg_network, state=sg_network.state)
        self.op_compute_gradients = self.Op(sg_gradients.calculate,
                                            state=sg_network.state, action=sg_loss.ph_action,
                                            discounted_reward=sg_loss.ph_discounted_reward)


if __name__ == '__main__':
    utils.assemble_and_show_graphs(SharedParameters, PolicyModel)
