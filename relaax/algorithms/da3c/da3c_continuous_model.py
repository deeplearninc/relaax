from __future__ import absolute_import

import logging
import numpy as np
import tensorflow as tf

from relaax.common.algorithms import subgraph
from relaax.common.algorithms.lib import graph
from relaax.common.algorithms.lib import layer
from relaax.common.algorithms.lib import loss
from relaax.common.algorithms.lib import optimizer
from relaax.common.algorithms.lib import lr_schedule
from . import da3c_config
from . import icm_model

logger = logging.getLogger(__name__)

tf.set_random_seed(da3c_config.config.seed)
np.random.seed(da3c_config.config.seed)


class Head(subgraph.Subgraph):
    def build_graph(self, input_placeholder):
        input = layer.ConfiguredInput(da3c_config.config.input, input_placeholder=input_placeholder)
        activation = layer.get_activation(da3c_config.config)

        sizes = da3c_config.config.hidden_sizes
        layers = [input]
        flattened_input = layer.Flatten(input)

        fc_layers = layer.GenericLayers(flattened_input, [dict(type=layer.Dense, size=size,
                                        activation=activation) for size in sizes[:-1]])
        layers.append(fc_layers)

        last_size = fc_layers.node.shape.as_list()[-1]
        if len(sizes) > 0:
            last_size = sizes[-1]

        if da3c_config.config.use_lstm:
            lstm = layer.lstm(da3c_config.config.lstm_type,
                              graph.Expand(fc_layers, 0), n_units=last_size,
                              n_cores=da3c_config.config.lstm_num_cores)
            head = graph.Reshape(lstm, [-1, last_size])
            layers.append(lstm)

            self.lstm_items = {"ph_lstm_state": lstm.ph_state,
                               "lstm_zero_state": lstm.zero_state,
                               "lstm_state": lstm.state,
                               "lstm_reset_timestep": lstm.reset_timestep}
        else:
            head = layer.Dense(fc_layers, last_size, activation=activation)
            layers.append(head)

        self.ph_state = input.ph_state
        self.weight = layer.Weights(*layers)
        return head.node


class Subnet(object):
    def __init__(self, head, weights,
                 ph_lstm_state=None, lstm_zero_state=None, lstm_state=None, lstm_reset_timestep=None):
        self.head = head
        self.weights = weights
        if da3c_config.config.use_lstm:
            self.ph_lstm_state = ph_lstm_state
            self.lstm_zero_state = lstm_zero_state
            self.lstm_state = lstm_state
            self.lstm_reset_timestep = lstm_reset_timestep


class Network(subgraph.Subgraph):
    def build_graph(self):
        input_placeholder = layer.InputPlaceholder(da3c_config.config.input)

        actor_head = Head(input_placeholder)
        actor = layer.Actor(actor_head, da3c_config.config.output)

        critic_head = Head(input_placeholder)
        critic = layer.Dense(critic_head, 1)

        self.ph_state = input_placeholder.ph_state

        feeds = dict(head=actor, weights=layer.Weights(actor_head, actor))
        if da3c_config.config.use_lstm:
            feeds.update(actor_head.lstm_items)
        self.actor = Subnet(**feeds)

        feeds = dict(head=graph.Flatten(critic), weights=layer.Weights(critic_head, critic))
        if da3c_config.config.use_lstm:
            feeds.update(critic_head.lstm_items)
        self.critic = Subnet(**feeds)

        self.weights = layer.Weights(actor_head, actor, critic_head, critic)
        if da3c_config.config.use_lstm:
            self.lstm_zero_state = (self.actor.lstm_zero_state, self.critic.lstm_zero_state)


# Weights of the policy are shared across
# all agents and stored on the parameter server
class SharedParameters(subgraph.Subgraph):
    def build_graph(self):
        # Build graph
        sg_global_step = graph.GlobalStep()
        sg_network = Network()
        self.actor = sg_network.actor
        self.critic = sg_network.critic

        if da3c_config.config.use_linear_schedule:
            sg_learning_rate = lr_schedule.Linear(sg_global_step, da3c_config.config)
        else:
            sg_learning_rate = da3c_config.config.initial_learning_rate

        if da3c_config.config.optimizer == 'Adam':
            sg_actor_optimizer = optimizer.AdamOptimizer(sg_learning_rate)
            sg_critic_optimizer = optimizer.AdamOptimizer(sg_learning_rate)
        else:
            sg_actor_optimizer = optimizer.RMSPropOptimizer(learning_rate=sg_learning_rate,
                                                            decay=da3c_config.config.RMSProp.decay,
                                                            momentum=0.0,
                                                            epsilon=da3c_config.config.RMSProp.epsilon)
            sg_critic_optimizer = optimizer.RMSPropOptimizer(learning_rate=sg_learning_rate,
                                                             decay=da3c_config.config.RMSProp.decay,
                                                             momentum=0.0,
                                                             epsilon=da3c_config.config.RMSProp.epsilon)

        sg_actor_gradients = optimizer.Gradients(self.actor.weights, optimizer=sg_actor_optimizer)
        sg_critic_gradients = optimizer.Gradients(self.critic.weights, optimizer=sg_critic_optimizer)

        if da3c_config.config.use_icm:
            sg_icm_optimizer = optimizer.AdamOptimizer(da3c_config.config.icm.lr)
            sg_icm_weights = icm_model.ICM().weights
            sg_icm_gradients = optimizer.Gradients(sg_icm_weights, optimizer=sg_icm_optimizer)

            # Expose ICM public API
            self.op_icm_get_weights = self.Op(sg_icm_weights)
            self.op_icm_apply_gradients = self.Op(sg_icm_gradients.apply,
                                                  gradients=sg_icm_gradients.ph_gradients)

        sg_average_reward = graph.LinearMovingAverage(da3c_config.config.avg_in_num_batches)
        sg_initialize = graph.Initialize()

        # Expose public API
        self.op_n_step = self.Op(sg_global_step.n)
        self.op_score = self.Op(sg_average_reward.average)

        self.op_check_weights = self.Ops(self.actor.weights.check, self.critic.weights.check)
        self.op_get_weights = self.Ops((self.actor.weights, self.critic.weights), sg_global_step.n)

        self.op_apply_gradients = self.Ops(sg_actor_gradients.apply, sg_critic_gradients.apply,
                                           sg_global_step.increment,
                                           gradients=(sg_actor_gradients.ph_gradients,
                                                      sg_critic_gradients.ph_gradients),
                                           increment=sg_global_step.ph_increment)
        self.op_add_rewards_to_model_score_routine = self.Ops(sg_average_reward.add,
                                                              reward_sum=sg_average_reward.ph_sum,
                                                              reward_weight=sg_average_reward.ph_count)

        # Determine Gradients' applying methods: fifo (by default), averaging, delay compensation
        sg_get_weights_flatten = graph.GetVariablesFlatten(sg_network.weights)
        sg_set_weights_flatten = graph.SetVariablesFlatten(sg_network.weights)
        self.op_get_weights_flatten = self.Op(sg_get_weights_flatten)
        self.op_set_weights_flatten = self.Op(sg_set_weights_flatten, value=sg_set_weights_flatten.ph_value)

        self.op_submit_gradients = self.Call(graph.get_gradients_apply_routine(da3c_config.config))

        self.op_initialize = self.Op(sg_initialize)


# Policy run by Agent(s)
class AgentModel(subgraph.Subgraph):
    def build_graph(self):
        # Build graph
        sg_network = Network()
        self.actor = sg_network.actor
        self.critic = sg_network.critic

        sg_loss = loss.DA3CLoss(sg_network.actor.head, sg_network.critic.head, da3c_config.config)
        sg_actor_gradients = optimizer.Gradients(sg_network.actor.weights,
                                                 loss=graph.TfNode(sg_loss.policy_loss),
                                                 norm=da3c_config.config.gradients_norm_clipping)
        sg_critic_gradients = optimizer.Gradients(sg_network.critic.weights,
                                                  loss=graph.TfNode(sg_loss.value_loss),
                                                  norm=da3c_config.config.gradients_norm_clipping)

        if da3c_config.config.use_icm:
            sg_icm_network = icm_model.ICM()
            sg_icm_gradients = optimizer.Gradients(sg_icm_network.weights, loss=sg_icm_network.loss)

            # Expose ICM public API
            self.op_icm_assign_weights = self.Op(sg_icm_network.weights.assign,
                                                 weights=sg_icm_network.weights.ph_weights)

            feeds = dict(state=sg_icm_network.ph_state, probs=sg_icm_network.ph_probs)
            self.op_get_intrinsic_reward = self.Ops(sg_icm_network.rew_out, **feeds)

            feeds.update(dict(action=sg_icm_network.ph_taken))
            self.op_compute_icm_gradients = self.Op(sg_icm_gradients.calculate, **feeds)

        summaries = tf.summary.merge([
            tf.summary.scalar('policy_loss', sg_loss.policy_loss),
            tf.summary.scalar('value_loss', sg_loss.value_loss),
            tf.summary.scalar('entropy', sg_loss.entropy),
            tf.summary.scalar('actor_gradients_global_norm', sg_actor_gradients.global_norm),
            tf.summary.scalar('critic_gradients_global_norm', sg_critic_gradients.global_norm),
            tf.summary.scalar('actor_weights_global_norm', sg_network.actor.weights.global_norm),
            tf.summary.scalar('critic_weights_global_norm', sg_network.critic.weights.global_norm)])

        # Expose public API
        self.op_assign_weights = self.Ops(sg_network.actor.weights.assign, sg_network.critic.weights.assign,
                                          weights=(sg_network.actor.weights.ph_weights,
                                                   sg_network.critic.weights.ph_weights))

        feeds = dict(state=sg_network.ph_state, action=sg_loss.ph_action,
                     advantage=sg_loss.ph_advantage, discounted_reward=sg_loss.ph_discounted_reward)

        if da3c_config.config.use_lstm:
            feeds.update(dict(lstm_state=(sg_network.actor.ph_lstm_state, sg_network.critic.ph_lstm_state)))
            self.lstm_zero_state = (sg_network.actor.lstm_zero_state, sg_network.critic.lstm_zero_state)
            self.op_lstm_reset_timestep = self.Ops(sg_network.actor.lstm_reset_timestep,
                                                   sg_network.critic.lstm_reset_timestep)
            self.op_get_action_value_and_lstm_state = \
                self.Ops(sg_network.actor.head, sg_network.critic.head,
                         (sg_network.actor.lstm_state, sg_network.critic.lstm_state),
                         state=sg_network.ph_state,
                         lstm_state=(sg_network.actor.ph_lstm_state, sg_network.critic.ph_lstm_state))
        else:
            self.op_get_action_and_value = self.Ops(sg_network.actor.head, sg_network.critic.head,
                                                    state=sg_network.ph_state)

        self.op_compute_gradients_and_summaries = \
            self.Ops((sg_actor_gradients.calculate, sg_critic_gradients.calculate), summaries, **feeds)
