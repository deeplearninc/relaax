from __future__ import absolute_import
import tensorflow as tf
import numpy as np

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

        actor = layer.DDPGActor(dense, cfg.config.output)

        self.ph_state = input.ph_state
        self.actor = actor
        self.weights = layer.Weights(input, dense, actor)


class CriticNetwork(subgraph.Subgraph):
    def build_graph(self):
        input = layer.Input(cfg.config.input)
        ph_action = graph.Placeholder(np.float32, (None, cfg.config.output.action_size))

        sizes = cfg.config.hidden_sizes
        assert len(sizes) > 1, 'You need to provide sizes at least for 2 layers'

        dense_1st = layer.Dense(layer.Flatten(input), sizes[0], layer.Activation.Relu)
        dense_2nd = layer.DoubleDense(dense_1st, ph_action, sizes[1], layer.Activation.Relu)

        critic = layer.Dense(dense_2nd, cfg.config.output.action_size)

        self.ph_state = input.ph_state
        self.ph_action = ph_action.node
        self.weights = layer.Weights(input, dense_1st, dense_2nd, critic)

        return graph.Flatten(critic).node


# Weights of the policy are shared across
# all agents and stored on the parameter server
class SharedParameters(subgraph.Subgraph):
    def build_graph(self):
        # Build graph
        sg_global_step = graph.GlobalStep()

        sg_actor_weights = ActorNetwork().weights
        sg_critic_weights = CriticNetwork().weights

        sg_actor_target_weights = ActorNetwork().weights
        sg_critic_target_weights = CriticNetwork().weights

        # needs reassign weights from actor & critic to target networks
        sg_init_actor_target_weights = \
            graph.TfNode([tf.assign(variable, value) for variable, value
                          in utils.Utils.izip(sg_actor_target_weights.node, sg_actor_weights.node)])
        sg_init_critic_target_weights = \
            graph.TfNode([tf.assign(variable, value) for variable, value
                          in utils.Utils.izip(sg_critic_target_weights.node, sg_critic_weights.node)])

        tau_res = graph.TfNode(1. - cfg.config.tau).node
        sg_update_actor_target_weights = \
            graph.TfNode([tf.assign(variable, tau_res * variable + cfg.config.tau * value) for variable, value
                          in utils.Utils.izip(sg_actor_target_weights.node, sg_actor_weights.node)])
        sg_update_critic_target_weights = \
            graph.TfNode([tf.assign(variable, tau_res * variable + cfg.config.tau * value) for variable, value
                          in utils.Utils.izip(sg_critic_target_weights.node, sg_critic_weights.node)])

        sg_actor_optimizer = graph.AdamOptimizer(cfg.config.actor_learning_rate)
        sg_critic_optimizer = graph.AdamOptimizer(cfg.config.critic_learning_rate)

        sg_actor_gradients = layer.Gradients(sg_actor_weights,
                                             optimizer=sg_actor_optimizer)
        sg_critic_gradients = layer.Gradients(sg_critic_weights,
                                              optimizer=sg_critic_optimizer)

        sg_initialize = graph.Initialize()

        # Expose public API
        self.op_n_step = self.Op(sg_global_step.n)

        self.op_get_actor_weights = self.Op(sg_actor_weights)
        self.op_get_critic_weights = self.Op(sg_critic_weights)
        self.op_get_actor_target_weights = self.Op(sg_actor_target_weights)
        self.op_get_critic_target_weights = self.Op(sg_critic_target_weights)

        self.op_init_actor_target_weights = self.Op(sg_init_actor_target_weights)
        self.op_init_critic_target_weights = self.Op(sg_init_critic_target_weights)

        self.op_update_actor_target_weights = self.Op(sg_update_actor_target_weights)
        self.op_update_critic_target_weights = self.Op(sg_update_critic_target_weights)

        self.op_apply_actor_gradients = self.Ops(sg_actor_gradients.apply,
                                                 sg_global_step.increment,
                                                 gradients=sg_actor_gradients.ph_gradients,
                                                 increment=sg_global_step.ph_increment)
        self.op_apply_critic_gradients = self.Ops(sg_critic_gradients.apply,
                                                  gradients=sg_critic_gradients.ph_gradients)
        self.op_initialize = self.Op(sg_initialize)


# Policy run by Agent(s)
class AgentModel(subgraph.Subgraph):
    def build_graph(self):
        # Build graph
        sg_actor_network = ActorNetwork()
        sg_critic_network = CriticNetwork()
        sg_actor_target_network = ActorNetwork()
        sg_critic_target_network = CriticNetwork()

        ph_action_gradient = graph.Placeholder(np.float32, (None, cfg.config.output.action_size))
        sg_actor_gradients = layer.Gradients(sg_actor_network.weights,
                                             loss=graph.TfNode(sg_actor_network.actor.scaled_out),
                                             grad_ys=-ph_action_gradient.node,
                                             batch_size=cfg.config.batch_size)

        sg_critic_sq_loss = loss.SquaredDiffLoss(sg_critic_network.node)
        sg_critic_l2loss = loss.L2Loss(sg_critic_network.weights, cfg.config.l2_decay)
        sg_critic_loss = graph.TfNode(sg_critic_sq_loss.node + sg_critic_l2loss.node)
        sg_critic_gradients = layer.Gradients(sg_critic_network.weights,
                                              loss=sg_critic_loss,
                                              batch_size=cfg.config.batch_size)
        sg_critic_action_gradients = layer.Gradients(graph.TfNode(sg_critic_network.ph_action),
                                                     loss=sg_critic_network,
                                                     batch_size=cfg.config.batch_size)

        # Expose public API
        self.op_assign_actor_weights = self.Op(sg_actor_network.weights.assign,
                                               weights=sg_actor_network.weights.ph_weights)
        self.op_assign_critic_weights = self.Op(sg_critic_network.weights.assign,
                                                weights=sg_critic_network.weights.ph_weights)
        self.op_assign_actor_target_weights = self.Op(sg_actor_target_network.weights.assign,
                                                      weights=sg_actor_target_network.weights.ph_weights)
        self.op_assign_critic_target_weights = self.Op(sg_critic_target_network.weights.assign,
                                                       weights=sg_critic_target_network.weights.ph_weights)

        self.op_get_action = self.Op(sg_actor_network.actor,  # needs scaled_out (2nd)
                                     state=sg_actor_network.ph_state)
        self.op_compute_actor_gradients = self.Op(sg_actor_gradients.calculate,
                                                  state=sg_actor_network.ph_state,
                                                  grad_ys=ph_action_gradient)

        self.op_get_value = self.Op(sg_critic_network,  # not used, cuz below try
                                    state=sg_critic_network.ph_state,
                                    action=sg_critic_network.ph_action)
        self.op_compute_critic_gradients = self.Op(sg_critic_gradients.calculate,
                                                   state=sg_critic_network.ph_state,
                                                   action=sg_critic_network.ph_action,
                                                   predicted=sg_critic_sq_loss.ph_predicted)

        self.op_compute_critic_action_gradients = self.Op(sg_critic_action_gradients.calculate,
                                                          state=sg_critic_network.ph_state,
                                                          action=sg_critic_network.ph_action)

        self.op_get_actor_target = self.Op(sg_actor_target_network.actor,  # needs scaled_out (2nd)
                                           state=sg_actor_target_network.ph_state)
        self.op_get_critic_target = self.Op(sg_critic_target_network,
                                            state=sg_critic_target_network.ph_state,
                                            action=sg_critic_target_network.ph_action)

if __name__ == '__main__':
    utils.assemble_and_show_graphs(SharedParameters, AgentModel)
