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
    def build_graph(self, n):
        self._n = n
        self.ph_sampled_variable = graph.TfNode(tf.placeholder(tf.int32, name='a'))

    class ProbVariableSubgraph(subgraph.Subgraph):
        def build_graph(self):
            return tf.placeholder(tf.float32, name='prob')

    class LoglikelihoodSubgraph(subgraph.Subgraph):
        def build_graph(self, prob, sampled_variable, n):
            return tf.log(tf.reduce_sum(prob.node *
                tf.one_hot(sampled_variable.node, n), axis=1))

    class KlSubgraph(subgraph.Subgraph):
        def build_graph(self, prob0, prob1):
            return tf.reduce_sum(prob0.node *
                tf.log(prob0.node / prob1.node), axis=1)

    class EntropySubgraph(subgraph.Subgraph):
        def build_graph(self, prob):
            return -tf.reduce_sum((prob.node * tf.log(prob).node), axis=1)

    @property
    def ProbVariable(self):
        return ProbVariableSubgraph

    @property
    def Loglikelihood(self):
        return lambda prob: LoglikelihoodSubgraph(
                prob, self.ph_sampled_variable, self._n)

    @property
    def Kl(self):
        return KlSubgraph

    @property
    def Entropy(self):
        return EntropySubgraph


class DiagGauss(subgraph.Subgraph):
    def build_graph(self, d):
        self._d = d
        self.ph_sampled_variable = graph.TfNode(tf.placeholder(tf.float32, name='a'))

    class ProbVariableSubgraph(subgraph.Subgraph):
        def build_graph(self):
            return tf.placeholder(tf.float32, name='prob')

    class LoglikelihoodSubgraph(subgraph.Subgraph):
        def build_graph(self, prob, sampled_variable, d):
            mean0 = prob.node[:, :d]
            std0 = prob.node[:, d:]
            return - 0.5 * tf.reduce_sum(tf.square((sampled_variable.node - mean0) /
                    std0), axis=1) - 0.5 * tf.log(2.0 * np.pi) * d - tf.reduce_sum(tf.log(std0), axis=1)

    class KlSubgraph(subgraph.Subgraph):
        def build_graph(self, prob0, prob1, d):
            mean0 = prob0.node[:, :d]
            std0 = prob0.node[:, d:]
            mean1 = prob1.node[:, :d]
            std1 = prob1.node[:, d:]
            return tf.reduce_sum(tf.log(std1 / std0), axis=1) + \
                    tf.reduce_sum(((tf.square(std0) + tf.square(mean0 - mean1)) /
                    (2.0 * tf.square(std1))), axis=1) - 0.5 * d

    class EntropySubgraph(subgraph.Subgraph):
        def build_graph(self, prob, d):
            std_nd = prob.node[:, d:]
            return tf.reduce_sum(tf.log(std_nd), axis=1) + .5 * np.log(2 * np.pi * np.e) * d

    @property
    def ProbVariable(self):
        return self.ProbVariableSubgraph

    @property
    def Loglikelihood(self):
        return lambda prob: self.LoglikelihoodSubgraph(
                prob, self.ph_sampled_variable, self._d)

    @property
    def Kl(self):
        return lambda prob0, prob1: self.KlSubgraph(prob0, prob1, self._d)

    @property
    def Entropy(self):
        return lambda prob: self.EntropySubgraph(prob, self._d)


def ProbType(*args):
    if trpo_config.config.output.continuous:
        return DiagGauss(*args)
    return Categorical(*args)


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

        ph_adv_n = graph.TfNode(tf.placeholder(tf.float32, name='adv_n'))

        sg_probtype = ProbType(trpo_config.config.output.action_size)

        ph_oldprob_np = sg_probtype.ProbVariable()

        sg_logp_n = sg_probtype.Loglikelihood(sg_network.actor)
        sg_oldlogp_n = sg_probtype.Loglikelihood(ph_oldprob_np)

        # Policy gradient:
        sg_surr = graph.TfNode(-tf.reduce_mean(tf.multiply(tf.exp(sg_logp_n.node -
            sg_oldlogp_n.node), ph_adv_n.node)))
        sg_gradients = layer.Gradients(sg_network.weights, loss=sg_surr)
        sg_gradients_flatten = GetVariablesFlatten(sg_gradients.calculate)

        sg_initialize = graph.Initialize()

        self.input = sg_network.ph_state
        self.output = sg_network.actor
        self.trainable_weights = list(utils.Utils.flatten(sg_network.weights.node))
        self.surr = sg_surr
        self.sampled_variable = sg_probtype.ph_sampled_variable
        self.prob_variable = ph_oldprob_np
        self.adv_n = ph_adv_n

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

        self.op_compute_policy_gradient = self.Op(sg_gradients_flatten,
                state=sg_network.ph_state, sampled_variable=self.sampled_variable,
                adv_n=ph_adv_n, oldprob_np=ph_oldprob_np)



# Policy run by Agent(s)
class AgentModel(subgraph.Subgraph):
    def build_graph(self):
        # Build graph
        sg_network = Network()

        self.input = sg_network.ph_state
        self.output = sg_network.actor
        # self.trainable_weights = list(utils.Utils.flatten(sg_network.weights.node))

        # Expose public API
        self.op_set_weights = self.Op(sg_network.weights.assign, weights=sg_network.weights.ph_weights)
        self.op_get_action = self.Op(sg_network.actor, state=sg_network.ph_state)


if __name__ == '__main__':
    utils.assemble_and_show_graphs(SharedParameters, AgentModel)
