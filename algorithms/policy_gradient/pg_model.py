from __future__ import absolute_import
import numpy as np

from relaax.common.algorithms import subgraph
from relaax.common.algorithms.lib import graph
from relaax.common.algorithms.lib import layer
from relaax.common.algorithms.lib import utils

from . import pg_config


class Network(subgraph.Subgraph):
    def build_graph(self):
        input = layer.Input(pg_config.config.input)

        weights = [input.weight]

        last = layer.Flatten(input)
        for size in pg_config.config.hidden_sizes:
            last = layer.Dense(last, size, activation=layer.Activation.Relu)
            weights.append(last.weight)

        last = layer.Dense(last, pg_config.config.action_size, activation=layer.Activation.Softmax)
        weights.append(last.weight)

        self.state = input.state
        self.weights = graph.Variables(*weights)
        return last.node


# Weights of the policy are shared across
# all agents and stored on the parameter server
class SharedParameters(subgraph.Subgraph):
    def build_graph(self):
        # Build graph
        ph_increment = graph.Placeholder(np.int64)
        sg_global_step = graph.GlobalStep(ph_increment)
        sg_weights = Network().weights
        ph_gradients = graph.Placeholders(variables=sg_weights)
        sg_optimizer = graph.AdamOptimizer(pg_config.config.learning_rate)
        sg_apply_gradients = graph.ApplyGradients(sg_optimizer, sg_weights, ph_gradients)
        sg_initialize = graph.Initialize()

        # Expose public API
        self.op_n_step = self.Op(sg_global_step.n)
        self.op_get_weights = self.Op(sg_weights)
        self.op_apply_gradients = self.Ops(sg_apply_gradients, sg_global_step.increment,
            gradients=ph_gradients,
            increment=ph_increment
        )
        self.op_initialize = self.Op(sg_initialize)


# Policy run by Agent(s)
class PolicyModel(subgraph.Subgraph):
    def build_graph(self):
        # Build graph
        sg_network = Network()
        ph_state = sg_network.state
        sg_weights = sg_network.weights
        ph_weights = graph.Placeholders(variables=sg_weights)
        sg_assign_weights = sg_weights.assign(ph_weights)

        ph_action = graph.Placeholder(np.int32, (None, ))
        ph_discounted_reward = graph.Placeholder(np.float32, (None, 1))

        sg_policy_loss = graph.PolicyLoss(
            action=ph_action,
            action_size=pg_config.config.action_size,
            discounted_reward=ph_discounted_reward,
            network=sg_network
        )

        sg_gradients = graph.Gradients(sg_policy_loss, sg_weights)

        # Expose public API
        self.op_assign_weights = self.Op(sg_assign_weights, weights=ph_weights)
        self.op_get_action = self.Op(sg_network, state=ph_state)
        self.op_compute_gradients = self.Op(sg_gradients,
            state=ph_state,
            action=ph_action,
            discounted_reward=ph_discounted_reward
        )


if __name__ == '__main__':
    utils.assemble_and_show_graphs(SharedParameters, PolicyModel)
