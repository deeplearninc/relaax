from __future__ import absolute_import

import logging
import numpy as np
import tensorflow as tf

from relaax.common.algorithms import subgraph
from relaax.common.algorithms.lib import graph
from relaax.common.algorithms.lib import layer
from relaax.common.algorithms.lib import optimizer
from relaax.common.algorithms.lib import utils
from relaax.common.algorithms.lib import lr_schedule

from . import dppo_config

from relaax.algorithms.trpo.trpo_model import (ProbType, ConcatFixedStd)

logger = logging.getLogger(__name__)

tf.set_random_seed(dppo_config.config.seed)
np.random.seed(dppo_config.config.seed)


class Network(subgraph.Subgraph):
    def build_graph(self, input_placeholder):
        input = layer.ConfiguredInput(dppo_config.config.input, input_placeholder=input_placeholder)
        layers = [input]

        sizes = dppo_config.config.hidden_sizes
        activation = layer.Activation.get_activation(dppo_config.config.activation)
        fc_layers = layer.GenericLayers(layer.Flatten(input),
                                        [dict(type=layer.Dense, size=size, activation=activation)
                                        for size in sizes[:-1]])
        layers.append(fc_layers)

        last_size = fc_layers.node.shape.as_list()[-1]
        if len(sizes) > 0:
            last_size = sizes[-1]

        if dppo_config.config.use_lstm:
            lstm = layer.LSTM(graph.Expand(fc_layers, 0), n_units=last_size)
            head = graph.Reshape(lstm, [-1, last_size])
            layers.append(lstm)

            self.ph_lstm_state = lstm.ph_state
            self.lstm_zero_state = lstm.zero_state
            self.lstm_state = lstm.state
        else:
            head = layer.Dense(fc_layers, last_size, activation=activation)
            layers.append(head)

        self.ph_state = input.ph_state
        self.weight = layer.Weights(*layers)
        return head.node


class Subnet(object):
    def __init__(self, head, weights, ph_state, ph_lstm_state=None, lstm_zero_state=None, lstm_state=None):
        self.head = head
        self.weights = weights
        self.ph_state = ph_state
        if dppo_config.config.use_lstm:
            self.ph_lstm_state = ph_lstm_state
            self.lstm_zero_state = lstm_zero_state
            self.lstm_state = lstm_state


class Model(subgraph.Subgraph):
    def build_graph(self, assemble_model=False):
        input_placeholder = layer.InputPlaceholder(dppo_config.config.input)

        policy_head = Network(input_placeholder)
        if dppo_config.config.output.continuous:
            output = layer.Dense(policy_head, dppo_config.config.output.action_size, init_var=0.01)
            actor = ConcatFixedStd(output)
            actor_layers = [output, actor]
        else:
            actor = layer.Dense(policy_head, dppo_config.config.output.action_size,
                                activation=layer.Activation.Softmax, init_var=0.01)
            actor_layers = [actor]

        value_head = Network(input_placeholder)
        critic = layer.Dense(value_head, 1)

        feeds = dict(head=actor, weights=layer.Weights(policy_head, *actor_layers),
                     ph_state=input_placeholder.ph_state)
        if dppo_config.config.use_lstm:
            feeds.update(dict(ph_lstm_state=policy_head.ph_lstm_state,
                              lstm_zero_state=policy_head.lstm_zero_state,
                              lstm_state=policy_head.lstm_state))
        self.actor = Subnet(**feeds)

        feeds = dict(head=critic, weights=layer.Weights(value_head, critic),
                     ph_state=input_placeholder.ph_state)
        if dppo_config.config.use_lstm:
            feeds.update(dict(ph_lstm_state=value_head.ph_lstm_state,
                              lstm_zero_state=value_head.lstm_zero_state,
                              lstm_state=value_head.lstm_state))
        self.critic = Subnet(**feeds)

        if assemble_model:
            self.policy = PolicyModel(self.actor)
            self.value_func = ValueModel(self.critic)

            if dppo_config.config.use_lstm:
                self.lstm_zero_state = [self.actor.lstm_zero_state, self.critic.lstm_zero_state]


class PolicyModel(subgraph.Subgraph):
    def build_graph(self, sg_network):
        if dppo_config.config.use_lstm:
            self.op_get_action = self.Ops(sg_network.head, sg_network.lstm_state,
                                          state=sg_network.ph_state, lstm_state=sg_network.ph_lstm_state)
            self.op_lstm_zero_state = sg_network.lstm_zero_state
        else:
            self.op_get_action = self.Op(sg_network.head, state=sg_network.ph_state)

        # Advantage node
        ph_adv_n = graph.TfNode(tf.placeholder(tf.float32, name='adv_n'))

        # Contains placeholder for the actual action made by the agent
        sg_probtype = ProbType(dppo_config.config.output.action_size,
                               continuous=dppo_config.config.output.continuous)

        # Placeholder to store action probabilities under the old policy
        ph_oldprob_np = sg_probtype.ProbVariable()

        sg_logp_n = sg_probtype.Loglikelihood(sg_network.head)
        sg_oldlogp_n = sg_probtype.Loglikelihood(ph_oldprob_np)

        # PPO clipped surrogate loss
        # likelihood ratio of old and new policy
        r_theta = tf.exp(sg_logp_n.node - sg_oldlogp_n.node)
        surr = r_theta * ph_adv_n.node
        clip_e = dppo_config.config.clip_e
        surr_clipped = tf.clip_by_value(r_theta, 1.0 - clip_e, 1.0 + clip_e) * ph_adv_n.node
        sg_pol_clip_loss = graph.TfNode(-tf.reduce_mean(tf.minimum(surr, surr_clipped)))

        # PPO entropy loss
        if dppo_config.config.entropy is not None:
            sg_entropy = sg_probtype.Entropy(sg_network.head)
            sg_ent_loss = (-dppo_config.config.entropy) * tf.reduce_mean(sg_entropy.node)
            sg_pol_total_loss = graph.TfNode(sg_pol_clip_loss.node + sg_ent_loss)
        else:
            sg_pol_total_loss = sg_pol_clip_loss

        # Regular gradients
        sg_ppo_clip_gradients = optimizer.Gradients(sg_network.weights, loss=sg_pol_total_loss,
                                                    norm=dppo_config.config.gradients_norm_clipping)
        feeds = dict(state=sg_network.ph_state, action=sg_probtype.ph_sampled_variable,
                     advantage=ph_adv_n, old_prob=ph_oldprob_np)
        if dppo_config.config.use_lstm:
            feeds.update(dict(lstm_state=sg_network.ph_lstm_state))

        self.op_compute_ppo_clip_gradients = self.Op(sg_ppo_clip_gradients.calculate, **feeds)
        if dppo_config.config.use_lstm:
            self.op_compute_ppo_clip_gradients = self.Ops(sg_ppo_clip_gradients.calculate,
                                                          sg_network.lstm_state, **feeds)

        # Weights get/set for updating the policy
        sg_get_weights_flatten = graph.GetVariablesFlatten(sg_network.weights)
        sg_set_weights_flatten = graph.SetVariablesFlatten(sg_network.weights)

        self.op_get_weights = self.Op(sg_network.weights)
        self.op_assign_weights = self.Op(sg_network.weights.assign,
                                         weights=sg_network.weights.ph_weights)

        self.op_get_weights_flatten = self.Op(sg_get_weights_flatten)
        self.op_set_weights_flatten = self.Op(sg_set_weights_flatten, value=sg_set_weights_flatten.ph_value)

        # Init Op for all weights
        sg_initialize = graph.Initialize()
        self.op_initialize = self.Op(sg_initialize)


# Value function model used by agents to estimate advantage
class ValueModel(subgraph.Subgraph):
    def build_graph(self, sg_value_net):
        # 'Observed' value of a state = discounted reward
        vf_scale = dppo_config.config.critic_scale

        ph_ytarg_ny = graph.Placeholder(np.float32)
        v1_loss = graph.TfNode(tf.square(sg_value_net.head.node - ph_ytarg_ny.node))

        if dppo_config.config.vf_clipped_loss:
            ph_old_vpred = graph.Placeholder(np.float32)
            clip_e = dppo_config.config.clip_e
            vpredclipped = ph_old_vpred.node + tf.clip_by_value(sg_value_net.head.node - ph_old_vpred.node,
                                                                -clip_e, clip_e)
            v2_loss = graph.TfNode(tf.square(vpredclipped - ph_ytarg_ny.node))
            vf_mse = graph.TfNode(vf_scale * tf.reduce_mean(tf.maximum(v2_loss.node, v1_loss.node)))
        else:
            vf_mse = graph.TfNode(vf_scale * tf.reduce_mean(v1_loss.node))

        if dppo_config.config.l2_coeff is not None:
            l2 = graph.TfNode(dppo_config.config.l2_coeff *
                              tf.add_n([tf.reduce_sum(tf.square(v)) for v in
                                        utils.Utils.flatten(sg_value_net.weights.node)]))

            sg_vf_total_loss = graph.TfNode(l2.node + vf_mse.node)
        else:
            sg_vf_total_loss = vf_mse

        sg_gradients = optimizer.Gradients(sg_value_net.weights, loss=sg_vf_total_loss,
                                           norm=dppo_config.config.gradients_norm_clipping)
        sg_gradients_flatten = graph.GetVariablesFlatten(sg_gradients.calculate)

        # Op to compute value of a state
        if dppo_config.config.use_lstm:
            self.op_value = self.Ops(sg_value_net.head, sg_value_net.lstm_state,
                                     state=sg_value_net.ph_state, lstm_state=sg_value_net.ph_lstm_state)
            self.op_lstm_zero_state = sg_value_net.lstm_zero_state
        else:
            self.op_value = self.Op(sg_value_net.head, state=sg_value_net.ph_state)

        self.op_get_weights = self.Op(sg_value_net.weights)
        self.op_assign_weights = self.Op(sg_value_net.weights.assign,
                                         weights=sg_value_net.weights.ph_weights)

        sg_get_weights_flatten = graph.GetVariablesFlatten(sg_value_net.weights)
        sg_set_weights_flatten = graph.SetVariablesFlatten(sg_value_net.weights)

        self.op_get_weights_flatten = self.Op(sg_get_weights_flatten)
        self.op_set_weights_flatten = self.Op(sg_set_weights_flatten, value=sg_set_weights_flatten.ph_value)

        feeds = dict(state=sg_value_net.ph_state, ytarg_ny=ph_ytarg_ny)
        if dppo_config.config.use_lstm:
            feeds.update(dict(lstm_state=sg_value_net.ph_lstm_state))
        if dppo_config.config.vf_clipped_loss:
            feeds.update(dict(vpred_old=ph_old_vpred))

        self.op_compute_gradients = self.Op(sg_gradients.calculate, **feeds)
        if dppo_config.config.use_lstm:
            self.op_compute_gradients = self.Ops(sg_gradients.calculate, sg_value_net.lstm_state, **feeds)

        self.op_compute_loss_and_gradient_flatten = self.Ops(sg_vf_total_loss, sg_gradients_flatten, **feeds)

        losses = [sg_vf_total_loss, vf_mse]
        if dppo_config.config.l2_coeff is not None:
            losses.append(l2)
        self.op_losses = self.Ops(*losses, **feeds)

        # Init Op for all weights
        sg_initialize = graph.Initialize()
        self.op_initialize = self.Op(sg_initialize)


# A generic subgraph to handle distributed weights updates
# Main public API for agents:
#   op_get_weights/op_get_weights_signed - returns current state of weights
#   op_submit_gradients - send new gradient to parameter server
# Ops to use on parameter server:
#    op_initialize - init all weights (including optimizer state)
class SharedWeights(subgraph.Subgraph):
    def build_graph(self, weights):
        # Build graph
        sg_global_step = graph.GlobalStep()
        sg_update_step = graph.GlobalStep()
        sg_weights = weights

        if dppo_config.config.schedule == 'linear':
            if dppo_config.config.schedule_step == 'update':
                sg_schedule_step = sg_update_step
            elif dppo_config.config.schedule_step == 'environment':
                sg_schedule_step = sg_global_step
            else:
                assert False, 'Valid options for the schedule step are: update OR environment.' \
                              'You provide the following option:'.format(dppo_config.config.schedule_step)
            sg_learning_rate = lr_schedule.Linear(sg_schedule_step,
                                                  dppo_config.config.learning_rate,
                                                  dppo_config.config.max_global_step)
        else:
            sg_learning_rate = dppo_config.config.learning_rate

        sg_optimizer = optimizer.AdamOptimizer(sg_learning_rate, epsilon=dppo_config.config.optimizer.epsilon)
        sg_gradients = optimizer.Gradients(sg_weights, optimizer=sg_optimizer)
        sg_average_reward = graph.LinearMovingAverage(dppo_config.config.avg_in_num_batches)
        sg_initialize = graph.Initialize()

        # Weights get/set for updating the policy
        sg_get_weights_flatten = graph.GetVariablesFlatten(sg_weights)
        sg_set_weights_flatten = graph.SetVariablesFlatten(sg_weights)

        # Expose public API
        self.op_n_step = self.Op(sg_global_step.n)
        self.op_upd_step = self.Op(sg_update_step.n)
        self.op_score = self.Op(sg_average_reward.average)

        self.op_inc_global_step = self.Ops(sg_global_step.increment, increment=sg_global_step.ph_increment)
        self.op_inc_global_step_and_average_reward = self.Ops(sg_global_step.increment,
                                                              sg_average_reward.add,
                                                              increment=sg_global_step.ph_increment,
                                                              reward_sum=sg_average_reward.ph_sum,
                                                              reward_weight=sg_average_reward.ph_count)

        self.op_get_weights = self.Op(sg_weights)
        self.op_get_weights_signed = self.Ops(sg_weights, sg_update_step.n)

        self.op_apply_gradients = self.Ops(sg_gradients.apply, sg_update_step.increment,
                                           gradients=sg_gradients.ph_gradients,
                                           increment=sg_update_step.ph_increment)

        self.op_get_weights_flatten = self.Op(sg_get_weights_flatten)
        self.op_set_weights_flatten = self.Op(sg_set_weights_flatten, value=sg_set_weights_flatten.ph_value)

        # Gradient combining routines
        self.op_submit_gradients = self.Call(graph.get_gradients_apply_routine(dppo_config.config))

        self.op_initialize = self.Op(sg_initialize)


# Weights of the policy are shared across
# all agents and stored on the parameter server
class SharedParameters(subgraph.Subgraph):
    def build_graph(self):
        sg_model = Model()

        sg_policy_shared = SharedWeights(sg_model.actor.weights)
        sg_value_func_shared = SharedWeights(sg_model.critic.weights)

        self.policy = sg_policy_shared
        self.value_func = sg_value_func_shared


if __name__ == '__main__':
    utils.assemble_and_show_graphs(SharedParameters, Model(assemble_model=True))
