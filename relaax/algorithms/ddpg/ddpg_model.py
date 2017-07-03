from __future__ import absolute_import

from relaax.common.algorithms import subgraph
from relaax.common.algorithms.lib import graph
from relaax.common.algorithms.lib import layer
from relaax.common.algorithms.lib import loss
from relaax.common.algorithms.lib import utils

from . import ddpg_config as cfg


class ActorNetwork(subgraph.Subgraph):
    def build_graph(self):
        input = layer.Input(cfg.config.input)

        dense = layer.GenericLayers(layer.Flatten(input),
                                    [dict(type=layer.Dense, size=size, activation=layer.Activation.Relu)
                                     for size in cfg.config.hidden_sizes])

        actor = layer.Dense(dense, cfg.config.output.action_size,
                            activation=layer.Activation.Tanh)

        self.state = input.ph_state
        self.weights = layer.Weights(input, dense, actor)
        return actor.node


class CriticNetwork(subgraph.Subgraph):
    def build_graph(self):
        input = layer.Input(cfg.config.input)

        dense = layer.GenericLayers(layer.Flatten(input),
                                    [dict(type=layer.Dense, size=size, activation=layer.Activation.Relu)
                                     for size in cfg.config.hidden_sizes])

        critic = layer.Dense(dense, cfg.config.output.action_size)

        self.state = input.ph_state
        self.weights = layer.Weights(input, dense, critic)
        return critic.node


# Weights of the policy are shared across
# all agents and stored on the parameter server
class SharedParameters(subgraph.Subgraph):
    def build_graph(self):
        # Build graph
        sg_global_step = graph.GlobalStep()

        sg_actor_weights = ActorNetwork().weights
        sg_critic_weights = CriticNetwork().weights

        sg_actor_optimizer = graph.AdamOptimizer(cfg.config.actor_learning_rate)
        sg_critic_optimizer = graph.AdamOptimizer(cfg.config.critic_learning_rate)

        sg_actor_gradients = layer.Gradients(sg_actor_weights, optimizer=sg_actor_optimizer)
        sg_critic_gradients = layer.Gradients(sg_critic_weights, optimizer=sg_critic_optimizer)

        sg_initialize = graph.Initialize()

        # Expose public API
        self.op_n_step = self.Op(sg_global_step.n)

        self.op_get_actor_weights = self.Op(sg_actor_weights)
        self.op_get_critic_weights = self.Op(sg_critic_weights)

        self.op_apply_actor_gradients = self.Ops(sg_actor_gradients.apply,
                                                 sg_global_step.increment,
                                                 gradients=sg_actor_gradients.ph_gradients,
                                                 increment=sg_global_step.ph_increment)
        self.op_apply_critic_gradients = self.Ops(sg_critic_gradients.apply,
                                                  sg_global_step.increment,
                                                  gradients=sg_critic_gradients.ph_gradients,
                                                  increment=sg_global_step.ph_increment)
        self.op_initialize = self.Op(sg_initialize)
