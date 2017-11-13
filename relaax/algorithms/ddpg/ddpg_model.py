from __future__ import absolute_import
import numpy as np

from relaax.common.algorithms import subgraph
from relaax.common.algorithms.lib import graph
from relaax.common.algorithms.lib import layer
from relaax.common.algorithms.lib import utils
from relaax.common.algorithms.lib import loss
from relaax.common.algorithms.lib import optimizer

from . import ddpg_config as cfg


class ActorNetwork(subgraph.Subgraph):
    def build_graph(self):
        input = layer.ConfiguredInput(cfg.config.input)

        dense = layer.GenericLayers(layer.Flatten(input),
                                    [dict(type=layer.Dense, size=size, activation=layer.Activation.Relu)
                                     for size in cfg.config.hidden_sizes])

        actor = layer.DDPGActor(dense, cfg.config.output)

        self.ph_state = input.ph_state
        self.actor = actor.scaled_out
        self.weights = layer.Weights(input, dense, actor)


class CriticNetwork(subgraph.Subgraph):
    def build_graph(self):
        input = layer.ConfiguredInput(cfg.config.input)
        self.ph_action = graph.Placeholder(np.float32, (None, cfg.config.output.action_size))

        sizes = cfg.config.hidden_sizes
        assert len(sizes) > 1, 'You need to provide sizes at least for 2 layers'

        dense_1st = layer.Dense(layer.Flatten(input), sizes[0], layer.Activation.Relu)
        dense_2nd = layer.DoubleDense(dense_1st, self.ph_action, sizes[1], layer.Activation.Relu)
        layers = [input, dense_1st, dense_2nd]

        net = layer.GenericLayers(dense_2nd,
                                  [dict(type=layer.Dense, size=size,
                                        activation=layer.Activation.Relu) for size in sizes[2:]])
        if len(sizes[2:]) > 0:
            layers.append(net)

        self.critic = layer.Dense(net, 1, init_var=3e-3)
        self.ph_state = input.ph_state

        layers.append(self.critic)
        self.weights = layer.Weights(*layers)


# Weights of the policy are shared across
# all agents and stored on the parameter server
class SharedParameters(subgraph.Subgraph):
    def build_graph(self):
        # Build graph
        sg_global_step = graph.GlobalStep()
        sg_episode_cnt = graph.GlobalStep()

        sg_actor_weights = ActorNetwork().weights
        sg_critic_weights = CriticNetwork().weights

        sg_actor_target_weights = ActorNetwork().weights
        sg_critic_target_weights = CriticNetwork().weights

        # needs reassign weights from actor & critic to target networks
        sg_init_actor_target_weights = \
            graph.AssignWeights(sg_actor_target_weights, sg_actor_weights).op
        sg_init_critic_target_weights = \
            graph.AssignWeights(sg_critic_target_weights, sg_critic_weights).op

        sg_update_actor_target_weights = \
            graph.AssignWeights(sg_actor_target_weights, sg_actor_weights, cfg.config.tau).op
        sg_update_critic_target_weights = \
            graph.AssignWeights(sg_critic_target_weights, sg_critic_weights, cfg.config.tau).op

        sg_actor_optimizer = optimizer.AdamOptimizer(cfg.config.actor_learning_rate)
        sg_critic_optimizer = optimizer.AdamOptimizer(cfg.config.critic_learning_rate)

        sg_actor_gradients = optimizer.Gradients(sg_actor_weights, optimizer=sg_actor_optimizer)
        sg_critic_gradients = optimizer.Gradients(sg_critic_weights, optimizer=sg_critic_optimizer)

        sg_initialize = graph.Initialize()

        # Expose public API
        self.op_get_weights = self.Ops(sg_actor_weights, sg_actor_target_weights,
                                       sg_critic_weights, sg_critic_target_weights)

        self.op_init_target_weights = self.Ops(sg_init_actor_target_weights,
                                               sg_init_critic_target_weights)

        self.op_update_target_weights = self.Ops(sg_update_actor_target_weights,
                                                 sg_update_critic_target_weights)

        self.op_apply_actor_gradients = self.Ops(sg_actor_gradients.apply, sg_global_step.increment,
                                                 gradients=sg_actor_gradients.ph_gradients,
                                                 increment=sg_global_step.ph_increment)
        self.op_apply_critic_gradients = self.Op(sg_critic_gradients.apply,
                                                 gradients=sg_critic_gradients.ph_gradients)

        self.op_n_step = self.Op(sg_global_step.n)
        self.op_inc_step = self.Op(sg_global_step.increment, increment=sg_global_step.ph_increment)

        self.op_get_episode_cnt = self.Op(sg_episode_cnt.n)
        self.op_inc_episode_cnt = self.Op(sg_episode_cnt.increment, increment=sg_episode_cnt.ph_increment)

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
        actor_grad_args = dict(loss=sg_actor_network.actor, grad_ys=-ph_action_gradient.node)

        if cfg.config.no_ps:
            sg_actor_optimizer = optimizer.AdamOptimizer(cfg.config.actor_learning_rate)
            actor_grad_args.update(dict(optimizer=sg_actor_optimizer))

        sg_actor_gradients = optimizer.Gradients(sg_actor_network.weights, **actor_grad_args)

        sg_critic_loss = loss.DDPGLoss(sg_critic_network, cfg.config)
        critic_grad_args = dict(loss=sg_critic_loss)

        if cfg.config.no_ps:
            sg_critic_optimizer = optimizer.AdamOptimizer(cfg.config.critic_learning_rate)
            critic_grad_args.update(dict(optimizer=sg_critic_optimizer))

        sg_critic_gradients = optimizer.Gradients(sg_critic_network.weights, **critic_grad_args)

        sg_critic_action_gradients = optimizer.Gradients(sg_critic_network.ph_action,
                                                     loss=sg_critic_network.critic)

        # Expose public API
        self.op_assign_actor_weights = self.Op(sg_actor_network.weights.assign,
                                               weights=sg_actor_network.weights.ph_weights)
        self.op_assign_critic_weights = self.Op(sg_critic_network.weights.assign,
                                                weights=sg_critic_network.weights.ph_weights)
        self.op_assign_actor_target_weights = self.Op(sg_actor_target_network.weights.assign,
                                                      weights=sg_actor_target_network.weights.ph_weights)
        self.op_assign_critic_target_weights = self.Op(sg_critic_target_network.weights.assign,
                                                       weights=sg_critic_target_network.weights.ph_weights)

        self.op_get_action = self.Op(sg_actor_network.actor,
                                     state=sg_actor_network.ph_state)
        self.op_get_critic_q = self.Op(sg_critic_network.critic,
                                       state=sg_critic_network.ph_state,
                                       action=sg_critic_network.ph_action)

        self.op_get_actor_target = self.Op(sg_actor_target_network.actor,
                                           state=sg_actor_target_network.ph_state)
        self.op_get_critic_target = self.Op(sg_critic_target_network.critic,
                                            state=sg_critic_target_network.ph_state,
                                            action=sg_critic_target_network.ph_action)

        self.op_compute_actor_gradients = self.Op(sg_actor_gradients.calculate,
                                                  state=sg_actor_network.ph_state,
                                                  grad_ys=ph_action_gradient)

        self.op_compute_critic_gradients = self.Op(sg_critic_gradients.calculate,
                                                   state=sg_critic_network.ph_state,
                                                   action=sg_critic_network.ph_action,
                                                   predicted=sg_critic_loss.ph_predicted)

        self.op_compute_critic_action_gradients = self.Op(sg_critic_action_gradients.calculate,
                                                          state=sg_critic_network.ph_state,
                                                          action=sg_critic_network.ph_action)

        # Integrated with grad computation by log_lvl
        self.op_critic_loss = self.Op(sg_critic_loss,
                                      state=sg_critic_network.ph_state,
                                      action=sg_critic_network.ph_action,
                                      predicted=sg_critic_loss.ph_predicted)
        self.op_compute_norm_actor_gradients = self.Op(sg_actor_gradients.global_norm,
                                                       state=sg_actor_network.ph_state,
                                                       grad_ys=ph_action_gradient)
        self.op_compute_norm_critic_gradients = self.Op(sg_critic_gradients.global_norm,
                                                        state=sg_critic_network.ph_state,
                                                        action=sg_critic_network.ph_action,
                                                        predicted=sg_critic_loss.ph_predicted)
        self.op_compute_norm_critic_action_gradients = self.Op(sg_critic_action_gradients.global_norm,
                                                               state=sg_critic_network.ph_state,
                                                               action=sg_critic_network.ph_action)

        if cfg.config.no_ps:
            sg_actor_weights = sg_actor_network.weights
            sg_critic_weights = sg_critic_network.weights

            sg_actor_target_weights = sg_actor_target_network.weights
            sg_critic_target_weights = sg_critic_target_network.weights

            # needs reassign weights from actor & critic to target networks
            sg_init_actor_target_weights = \
                graph.AssignWeights(sg_actor_target_weights, sg_actor_weights).op
            sg_init_critic_target_weights = \
                graph.AssignWeights(sg_critic_target_weights, sg_critic_weights).op

            sg_update_actor_target_weights = \
                graph.AssignWeights(sg_actor_target_weights, sg_actor_weights, cfg.config.tau).op
            sg_update_critic_target_weights = \
                graph.AssignWeights(sg_critic_target_weights, sg_critic_weights, cfg.config.tau).op

            self.op_get_weights = self.Ops(sg_actor_weights, sg_actor_target_weights,
                                           sg_critic_weights, sg_critic_target_weights)

            self.op_init_target_weights = self.Ops(sg_init_actor_target_weights,
                                                   sg_init_critic_target_weights)

            self.op_update_target_weights = self.Ops(sg_update_actor_target_weights,
                                                     sg_update_critic_target_weights)

            self.op_apply_actor_gradients = self.Ops(sg_actor_gradients.apply,
                                                     gradients=sg_actor_gradients.ph_gradients)
            self.op_apply_critic_gradients = self.Op(sg_critic_gradients.apply,
                                                     gradients=sg_critic_gradients.ph_gradients)
            sg_initialize = graph.Initialize()
            self.op_initialize = self.Op(sg_initialize)


if __name__ == '__main__':
    utils.assemble_and_show_graphs(SharedParameters, AgentModel)
