from __future__ import absolute_import
import numpy as np

from relaax.common.algorithms import subgraph
from relaax.common.algorithms.lib import graph
from relaax.common.algorithms.lib import layer
from relaax.common.algorithms.lib import loss
from relaax.common.algorithms.lib import utils
from .lib import da3c_graph
from . import da3c_config


class Network(subgraph.Subgraph):
    def build_graph(self):
        input = layer.Input(da3c_config.config.input)

        if da3c_config.config.input.use_convolutions:
            head = layer.Dense(layer.Flatten(input), size=256,
                    activation=layer.Activation.Relu)
        else:
            head = layer.GenericLayers(input, [
                    dict(type=Dense, size=300, activation=layer.Activation.Relu),
                    dict(type=Dense, size=200, activation=layer.Activation.Relu),
                    dict(type=Dense, size=100, activation=layer.Activation.Relu)])

        actor = layer.Actor(head, da3c_config.config.output)
        critic = layer.Dense(head, 1)

        self.ph_state = input.ph_state
        self.actor = actor
        self.critic = graph.Flatten(critic)
        self.weights = layer.Weights(input, head, actor, critic)


# Weights of the policy are shared across
# all agents and stored on the parameter server
class SharedParameters(subgraph.Subgraph):
    def build_graph(self):
        # Build graph
        sg_global_step = graph.GlobalStep()
        sg_weights = Network().weights
        sg_learning_rate = da3c_graph.LearningRate(sg_global_step)
        sg_optimizer = graph.RMSPropOptimizer(
            learning_rate=sg_learning_rate,
            decay=da3c_config.config.RMSProp.decay,
            momentum=0.0,
            epsilon=da3c_config.config.RMSProp.epsilon
        )
        sg_gradients = layer.Gradients(sg_weights, optimizer=sg_optimizer)
        sg_initialize = graph.Initialize()

        # Expose public API
        self.op_n_step = self.Op(sg_global_step.n)
        self.op_get_weights = self.Op(sg_weights)
        self.op_apply_gradients = self.Ops(sg_gradients.apply,
                sg_global_step.increment, gradients=sg_gradients.ph_gradients,
                increment=sg_global_step.ph_increment)
        self.op_initialize = self.Op(sg_initialize)


# Policy run by Agent(s)
class AgentModel(subgraph.Subgraph):
    def build_graph(self):
        # Build graph
        sg_network = Network()

        sg_loss = loss.DA3CLoss(sg_network.actor, sg_network.critic,
                da3c_config.config.entropy_beta)
        sg_gradients = layer.Gradients(sg_network.weights, loss=sg_loss)

        # Expose public API
        self.op_assign_weights = self.Op(sg_network.weights.assign,
                weights=sg_network.weights.ph_weights)
        self.op_get_action_and_value = self.Ops(
                sg_network.actor, sg_network.critic, state=sg_network.ph_state)
        self.op_compute_gradients = self.Op(sg_gradients.calculate,
                state=sg_network.ph_state, action=sg_loss.ph_action,
                value=sg_loss.ph_value, discounted_reward=sg_loss.ph_discounted_reward)


if __name__ == '__main__':
    utils.assemble_and_show_graphs(SharedParameters, AgentModel)
