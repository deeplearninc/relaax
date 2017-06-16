from __future__ import absolute_import

import numpy as np
import tensorflow as tf

from relaax.common.algorithms import subgraph
from relaax.common.algorithms.lib import graph
from relaax.common.algorithms.lib import layer
from relaax.common.algorithms.lib import loss
from relaax.common.algorithms.lib import utils

from . import trpo_config
from .lib import trpo_graph


class GetVariablesFlatten(subgraph.Subgraph):
    def build_graph(self, variables):
        tensors = utils.Utils.flatten(variables.node)
        return tf.concat([tf.reshape(t, [np.prod(t.shape.as_list())])
            for t in tensors], axis=0)


class SetVariablesFlatten(subgraph.Subgraph):
    def build_graph(self, variables):
        tensors = list(utils.Utils.flatten(variables.node))
        total_size = sum(np.prod(t.shape.as_list()) for t in tensors)
        self.ph_value = tf.placeholder(tf.float32, [total_size])
        start = 0
        assignes = []
        for t in tensors:
            shape = t.shape.as_list()
            size = np.prod(shape)
            assignes.append(tf.assign(t,
                tf.reshape(self.ph_value[start:start + size], shape)))
            start += size
        assert start == total_size
        return tf.group(*assignes)


class Categorical(subgraph.Subgraph):
    def build_graph(self, n, a, prob):
        self.ph_sampled_variable = graph.TfNode(tf.placeholder(tf.int32))
        self.ph_prob_variable = graph.TfNode(tf.placeholder(tf.float32))
        self.likelihood = tf.reduce_sum(prob.node * tf.one_hot(a.node, n), axis=1)
        self.loglikelihood = tf.log(self.likelihood)

    def kl(self, prob0, prob1):
        return tf.reduce_sum((prob0 * tf.log(prob0 / prob1)), axis=1)

    def entropy(self, prob0):
        return -tf.reduce_sum((prob0 * tf.log(prob0)), axis=1)


class DiagGauss(subgraph.Subgraph):
    def build_graph(self, d, a, prob):
        mean0 = prob.node[:, :self.d]
        std0 = prob.node[:, self.d:]

        self.ph_sampled_variable = graph.TfNode(tf.placeholder(tf.float32))
        self.ph_prob_variable = graph.TfNode(tf.placeholder(tf.float32))

        self.loglikelihood = - 0.5 * tf.reduce_sum(tf.square((a.node - mean0) / std0), axis=1) \
               - 0.5 * tf.log(2.0 * np.pi) * d - tf.reduce_sum(tf.log(std0), axis=1)
        self.likelihood = tf.exp(self.loglikelihood)

    def kl(self, prob0, prob1):
        mean0 = prob0[:, :self.d]
        std0 = prob0[:, self.d:]
        mean1 = prob1[:, :self.d]
        std1 = prob1[:, self.d:]
        return tf.reduce_sum(tf.log(std1 / std0), axis=1) + tf.reduce_sum(
            ((tf.square(std0) + tf.square(mean0 - mean1)) / (2.0 * tf.square(std1))), axis=1) - 0.5 * self.d

    def entropy(self, prob):
        std_nd = prob[:, self.d:]
        return tf.reduce_sum(tf.log(std_nd), axis=1) + .5 * np.log(2 * np.pi * np.e) * self.d


class ConcatFixedStd(subgraph.Subgraph):
    def build_graph(self, x):
        input_dim = x.node.get_shape().as_list()[1]
        logstd = tf.Variable(tf.zeros(input_dim, tf.float32))

        std = tf.tile(tf.reshape(tf.exp(logstd), [1, -1]), (tf.shape(x.node)[0], 1))

        self.weight = graph.TfNode(logstd)
        return tf.concat([x.node, std], axis=1)


class Network(subgraph.Subgraph):
    def build_graph(self):
        input = layer.Input(trpo_config.config.input)

        head = layer.GenericLayers(layer.Flatten(input),
                [dict(type=layer.Dense, size=size, activation=layer.Activation.Tanh)
                for size in trpo_config.config.hidden_sizes])

        if trpo_config.config.output.continuous:
            output = layer.Dense(head, trpo_config.config.output.action_size)
            actor = ConcatFixedStd(output)
            actor_layers = [output, actor]
        else:
            actor = layer.Dense(head, trpo_config.config.output.action_size,
                    activation=layer.Activation.Softmax)
            actor_layers = [actor]

        self.ph_state = input.ph_state
        self.actor = actor
        self.weights = layer.Weights(*([input, head] + actor_layers))



# Weights of the policy are shared across
# all agents and stored on the parameter server
class SharedParameters(subgraph.Subgraph):
    def wait_for_iteration(self, session):
        return self._ps_bridge.wait_for_iteration()

    def send_experience(self, session, n_iter, paths, length):
        self._ps_bridge.send_experience(n_iter, paths, length)

    def receive_weights(self, session, n_iter):
        return self._ps_bridge.receive_weights(n_iter)

    def build_graph(self):
        # Build graph
        
        sg_n_iter = trpo_graph.NIter()

        sg_global_step = graph.GlobalStep()

        sg_network = Network()
        sg_get_weights_flatten = GetVariablesFlatten(sg_network.weights)
        sg_set_weights_flatten = SetVariablesFlatten(sg_network.weights)
        #sg_gradients = layer.Gradients(loss, sg_network.weights)

        sg_initialize = graph.Initialize()

        self.input = sg_network.ph_state
        self.output = sg_network.actor
        self.trainable_weights = list(utils.Utils.flatten(sg_network.weights.node))

        # Expose public API
        self.op_n_step = self.Op(sg_global_step.n)
        self.op_inc_step = self.Op(sg_global_step.increment, increment=sg_global_step.ph_increment)
        self.op_initialize = self.Op(sg_initialize)

        self.call_wait_for_iteration = self.Call(self.wait_for_iteration)
        self.call_send_experience = self.Call(self.send_experience)
        self.call_receive_weights = self.Call(self.receive_weights)

        self.op_turn_collect_on = sg_n_iter.op_turn_collect_on
        self.op_turn_collect_off = sg_n_iter.op_turn_collect_off
        self.op_n_iter_value = sg_n_iter.op_n_iter_value
        self.op_n_iter = sg_n_iter.op_n_iter
        self.op_next_iter = sg_n_iter.op_next_iter

        self.op_get_action = self.Op(sg_network.actor, state=sg_network.ph_state)

        self.op_get_weights_flatten = self.Op(sg_get_weights_flatten)
        self.op_set_weights_flatten = self.Op(sg_set_weights_flatten, weights=sg_set_weights_flatten.ph_value)

        self.op_get_weights = self.Op(sg_network.weights)


# Policy run by Agent(s)
class AgentModel(subgraph.Subgraph):
    def build_graph(self):
        # Build graph
        sg_network = Network()

        self.input = sg_network.ph_state
        self.output = sg_network.actor
        self.trainable_weights = list(utils.Utils.flatten(sg_network.weights.node))

        # Expose public API
        self.op_set_weights = self.Op(sg_network.weights.assign, weights=sg_network.weights.ph_weights)
        self.op_get_action = self.Op(sg_network.actor, state=sg_network.ph_state)


if __name__ == '__main__':
    utils.assemble_and_show_graphs(SharedParameters, AgentModel)
