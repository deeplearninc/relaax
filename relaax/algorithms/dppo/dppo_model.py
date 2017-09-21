from __future__ import absolute_import

import logging
import numpy as np
import tensorflow as tf

from relaax.common.algorithms import subgraph
from relaax.common.algorithms.lib import graph
from relaax.common.algorithms.lib import layer
from relaax.common.algorithms.lib import optimizer
from relaax.common.algorithms.lib import loss
from relaax.common.algorithms.lib import utils

from . import dppo_config

from relaax.algorithms.trpo.trpo_model import GetVariablesFlatten, SetVariablesFlatten, Categorical, ProbType

logger = logging.getLogger(__name__)


class Network(subgraph.Subgraph):
    def build_graph(self):
        input = layer.Input(dppo_config.config.input)

        dense = layer.GenericLayers(layer.Flatten(input),
                                    [dict(type=layer.Dense, size=size, activation=layer.Activation.Relu)
                                     for size in dppo_config.config.hidden_sizes])

        actor = layer.Dense(dense, dppo_config.config.output.action_size,
                            activation=layer.Activation.Softmax)

        self.state = input.ph_state
        self.actor = actor
        self.weights = layer.Weights(input, dense, actor)
        return actor.node


class PolicyModel(subgraph.Subgraph):
    def build_graph(self):
        sg_network = Network()

        self.op_get_action = self.Op(sg_network, state=sg_network.state)

        # Advantage node
        ph_adv_n = graph.TfNode(tf.placeholder(tf.float32, name='adv_n'))

        # Contains placeholder for the actual action made by the agent
        sg_probtype = ProbType(dppo_config.config.output.action_size)

        # Placeholder to store action probabilities under the old policy
        ph_oldprob_np = sg_probtype.ProbVariable()

        sg_logp_n = sg_probtype.Loglikelihood(sg_network.actor)
        sg_oldlogp_n = sg_probtype.Loglikelihood(ph_oldprob_np)

        # PPO clipped surrogate loss
        # likelihood ratio of old and new policy
        r_theta = tf.exp(sg_logp_n.node - sg_oldlogp_n.node)
        surr = r_theta * ph_adv_n.node
        clip_e = dppo_config.config.clip_e
        surr_clipped = tf.clip_by_value(r_theta, 1.0 - clip_e, 1.0 + clip_e) * ph_adv_n.node
        sg_ppo_clip_loss = graph.TfNode(-tf.reduce_mean(tf.minimum(surr, surr_clipped)))

        # Regular gradients
        sg_ppo_clip_gradients = optimizer.Gradients(sg_network.weights,
                                                    loss=sg_ppo_clip_loss)
        self.op_compute_ppo_clip_gradients = self.Op(sg_ppo_clip_gradients.calculate,
                                                     state=sg_network.state,
                                                     action=sg_probtype.ph_sampled_variable,
                                                     advantage=ph_adv_n,
                                                     old_prob=ph_oldprob_np)

        # Flattened gradients
        sg_ppo_clip_gradients_flatten = GetVariablesFlatten(sg_ppo_clip_gradients.calculate)

        # Weights get/set for updating the policy
        sg_get_weights_flatten = GetVariablesFlatten(sg_network.weights)
        sg_set_weights_flatten = SetVariablesFlatten(sg_network.weights)

        self.op_get_weights = self.Op(sg_network.weights)
        self.op_assign_weights = self.Op(sg_network.weights.assign,
                                         weights=sg_network.weights.ph_weights)

        self.op_get_weights_flatten = self.Op(sg_get_weights_flatten)
        self.op_set_weights_flatten = self.Op(sg_set_weights_flatten, value=sg_set_weights_flatten.ph_value)

        # Init Op for all weights
        sg_initialize = graph.Initialize()
        self.op_initialize = self.Op(sg_initialize)


class ValueNet(subgraph.Subgraph):
    def build_graph(self):

        input = layer.Input(dppo_config.config.input)

        activation = layer.Activation.get_activation(dppo_config.config.activation)
        descs = [dict(type=layer.Dense, size=size, activation=activation) for size
                 in dppo_config.config.hidden_sizes]
        descs.append(dict(type=layer.Dense, size=1))

        value = layer.GenericLayers(layer.Flatten(input), descs)

        weights = layer.Weights(input, value)

        self.ph_state = input.ph_state
        self.weights = weights
        self.value = value

        return value.node


# Value function model used by agents to estimate advantage
class ValueModel(subgraph.Subgraph):
    def build_graph(self):
        sg_value_net = ValueNet()

        # 'Observed' value of a state = discounted reward
        ph_ytarg_ny = graph.Placeholder(np.float32)

        mse = graph.TfNode(tf.reduce_mean(tf.square(ph_ytarg_ny.node - sg_value_net.value.node)))

        logger.info("ValueModel | mse={}".format(mse.node))

        l2 = graph.TfNode(1e-3 * tf.add_n([tf.reduce_sum(tf.square(v)) for v in
                                           utils.Utils.flatten(sg_value_net.weights.node)]))

        logger.info("ValueModel | l2={}".format(l2.node))

        loss = graph.TfNode(l2.node + mse.node)

        sg_gradients = optimizer.Gradients(sg_value_net.weights, loss=loss)
        sg_gradients_flatten = GetVariablesFlatten(sg_gradients.calculate)

        # Op to compute value of a state
        self.op_value = self.Op(sg_value_net.value, state=sg_value_net.ph_state)

        self.op_get_weights = self.Op(sg_value_net.weights)
        self.op_assign_weights = self.Op(sg_value_net.weights.assign,
                                         weights=sg_value_net.weights.ph_weights)

        sg_get_weights_flatten = GetVariablesFlatten(sg_value_net.weights)
        sg_set_weights_flatten = SetVariablesFlatten(sg_value_net.weights)

        self.op_get_weights_flatten = self.Op(sg_get_weights_flatten)
        self.op_set_weights_flatten = self.Op(sg_set_weights_flatten, value=sg_set_weights_flatten.ph_value)

        self.op_compute_gradients = self.Op(sg_gradients.calculate, state=sg_value_net.ph_state,
                                             ytarg_ny=ph_ytarg_ny)

        self.op_compute_loss_and_gradient_flatten = self.Ops(loss, sg_gradients_flatten, state=sg_value_net.ph_state,
                                                             ytarg_ny=ph_ytarg_ny)

        self.op_losses = self.Ops(loss, mse, l2, state=sg_value_net.ph_state, ytarg_ny=ph_ytarg_ny)


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
        sg_weights = weights
        sg_optimizer = optimizer.AdamOptimizer(dppo_config.config.learning_rate)
        sg_gradients = optimizer.Gradients(sg_weights, optimizer=sg_optimizer)
        sg_initialize = graph.Initialize()

        # Weights get/set for updating the policy
        sg_get_weights_flatten = GetVariablesFlatten(sg_weights)
        sg_set_weights_flatten = SetVariablesFlatten(sg_weights)

        # Expose public API
        self.op_n_step = self.Op(sg_global_step.n)
        self.op_get_weights = self.Op(sg_weights)
        self.op_get_weights_signed = self.Ops(sg_weights, sg_global_step.n)

        self.op_apply_gradients = self.Ops(sg_gradients.apply,
                                           sg_global_step.increment, gradients=sg_gradients.ph_gradients,
                                           increment=sg_global_step.ph_increment)

        self.op_get_weights_flatten = self.Op(sg_get_weights_flatten)
        self.op_set_weights_flatten = self.Op(sg_set_weights_flatten, value=sg_set_weights_flatten.ph_value)

        # Gradient combining routines

        # First come, first served gradient update
        def func_fifo_gradient(session, gradients, step):
            current_step = session.op_n_step()
            logger.debug("Gradient with step {} received from agent. Current step: {}".format(step, current_step))
            session.op_apply_gradients(gradients=gradients, increment=1)

        # Accumulate gradients from many agents and average them
        self.gradients = []

        def func_average_gradient(session, gradients, step):
            logger.debug("received a gradient, number of gradients collected so far: {}".format(len(self.gradients)))
            if step >= session.op_n_step():
                logger.debug("gradient is fresh, accepted")
                self.gradients.append(gradients)
            else:
                logger.debug("gradient is old, rejected")

            if len(self.gradients) >= dppo_config.config.num_gradients:
                # we have collected enough gradients, no we can average them and make a step
                logger.debug("computing mean grad")
                flat_grads = [Shaper.get_flat(g) for g in self.gradients]
                mean_flat_grad = np.mean(np.stack(flat_grads), axis=0)

                mean_grad = Shaper.reverse(mean_flat_grad, gradients)
                session.op_apply_gradients(gradients=mean_grad, increment=1)
                self.gradients = []

        # Asynchronous Stochastic Gradient Descent with Delay Compensation, see https://arxiv.org/pdf/1609.08326.pdf
        self.weights_history = {}

        def init_weight_history(session):
            self.weights_history[0] = session.op_get_weights_flatten()

        self.op_init_weight_history = self.Call(init_weight_history)

        def func_dc_gradient(session, gradients, step):
            #logger.debug("session = {}, {}".format(session._name, session._full_path()))
            # Assume step to be global step number
            #current_step = session.op_n_step()

            #current_weights = session.op_get_weights()
            current_weights_f = session.op_get_weights_flatten()

            old_weights_f = self.weights_history.get(step, current_weights_f)

            new_gradient_f = Shaper.get_flat(gradients)

            # Compute new gradient
            delta = dppo_config.config.dc_lambda * (
                new_gradient_f * new_gradient_f * (current_weights_f - old_weights_f))
            compensated_gradient_f = new_gradient_f + delta

            compensated_gradient = Shaper.reverse(compensated_gradient_f, gradients)

            session.op_apply_gradients(gradients=compensated_gradient, increment=1)
            updated_weights = session.op_get_weights_flatten()
            updated_step = session.op_n_step()
            self.weights_history[updated_step] = updated_weights

            # Cleanup history
            for k in list(self.weights_history.keys()):
                if k < updated_step - 20:
                    try:
                        del self.weights_history[k]
                    except KeyError:
                        pass

        if dppo_config.config.combine_gradient == 'fifo':
            self.op_submit_gradients = self.Call(func_fifo_gradient)
        elif dppo_config.config.combine_gradient == 'average':
            self.op_submit_gradients = self.Call(func_average_gradient)
        elif dppo_config.config.combine_gradient == 'dc':
            self.op_submit_gradients = self.Call(func_dc_gradient)
        else:
            logger.error("Unknown gradient combination mode: {}".format(dppo_config.config.combine_gradient))

        self.op_initialize = self.Op(sg_initialize)


# Weights of the policy are shared across
# all agents and stored on the parameter server
class SharedParameters(subgraph.Subgraph):
    def build_graph(self):
        sg_policy = Network()
        sg_value_func = ValueNet()

        sg_policy_shared = SharedWeights(sg_policy.weights)
        sg_value_func_shared = SharedWeights(sg_value_func.weights)

        self.policy = sg_policy_shared
        self.value_func = sg_value_func_shared


class Shaper():
    @staticmethod
    def numel(x):
        return np.prod(np.shape(x))

    @classmethod
    def get_flat(cls, v):
        tensor_list = list(utils.Utils.flatten(v))
        u = np.concatenate([np.reshape(t, newshape=[cls.numel(t), ]) for t in tensor_list], axis=0)
        return u

    @staticmethod
    def reverse(u, base_shape):
        tensor_list = list(utils.Utils.flatten(base_shape))
        shapes = map(np.shape, tensor_list)
        v_flat = []
        start = 0
        for (shape, t) in zip(shapes, tensor_list):
            size = np.prod(shape)
            v_flat.append(np.reshape(u[start:start + size], shape))
            start += size
        v = utils.Utils.reconstruct(v_flat, base_shape)
        return v


if __name__ == '__main__':
    utils.assemble_and_show_graphs(SharedParameters, PolicyModel)
