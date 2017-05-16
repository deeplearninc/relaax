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

        layers = [input]

        last = layer.Flatten(input)
        for size in pg_config.config.hidden_sizes:
            last = layer.Dense(last, size, activation=layer.Activation.Relu)
            layers.append(last)

        last = layer.Dense(last, pg_config.config.action_size, activation=layer.Activation.Softmax)
        layers.append(last)

        self.state = input.state
        self.weights = layer.Weigths(*layers)
        return last.node


# Weights of the policy are shared across
# all agents and stored on the parameter server
class SharedParameters(subgraph.Subgraph):
    def build_graph(self):
        # Build graph
        sg_global_step = graph.GlobalStep()
        sg_weights = Network().weights
        sg_optimizer = graph.AdamOptimizer(pg_config.config.learning_rate)
        sg_gradients = layer.Gradients(sg_weights, optimizer=sg_optimizer)
        sg_initialize = graph.Initialize()

        # Expose public API
        self.op_n_step = self.Op(sg_global_step.n)
        self.op_get_weights = self.Op(sg_weights)
        self.op_apply_gradients = self.Ops(sg_gradients.apply,
                sg_global_step.increment, gradients=sg_gradients.placeholders,
                increment=sg_global_step.placeholder)
        self.op_initialize = self.Op(sg_initialize)


# Policy run by Agent(s)
class PolicyModel(subgraph.Subgraph):
    def build_graph(self):
        # Build graph
        sg_network = Network()

        sg_loss = graph.PolicyLoss(action_size=pg_config.config.action_size,
                network=sg_network)
        sg_gradients = layer.Gradients(sg_network.weights, loss=sg_loss)

        # Expose public API
        self.op_assign_weights = self.Op(sg_network.weights.assign,
                weights=sg_network.weights.placeholders)
        self.op_get_action = self.Op(sg_network, state=sg_network.state)
        self.op_compute_gradients = self.Op(sg_gradients.calculate,
                state=sg_network.state, action=sg_loss.action,
                discounted_reward=sg_loss.discounted_reward)


if __name__ == '__main__':
    utils.assemble_and_show_graphs(SharedParameters, PolicyModel)
