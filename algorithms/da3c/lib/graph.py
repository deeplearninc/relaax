import itertools
import numpy as np
import tensorflow as tf

from relaax.common.algorithms import subgraph
from relaax.common.algorithms.lib import graph

from .. import da3c_config


Initialize = graph.Initialize


class Counter(subgraph.Subgraph):
    DTYPE = {np.int64: tf.int64}

    def build_graph(self, inc_value, dtype=np.int64):
        counter = tf.Variable(0, dtype=self.DTYPE[dtype])
        increment = tf.assign_add(counter, inc_value.node)
        return {
            'value': counter,
            'increment': increment
        }

    def value(self):
        return subgraph.Subgraph.Op(self.node['value'])


class Weights(subgraph.Subgraph):
    def build_graph(self):
        return {
            'conv1': self._vars(16, 8, 8, 4),  # stride=4
            'conv2': self._vars(32, 4, 4, 16),  # stride=2

            'fc1': self._vars(256, 2592),

            # weight for policy output layer
            'fc2': self._vars(da3c_config.config.action_size, 256),

            # weight for value output layer
            'fc3': self._vars(1, 256)
        }

    def get(self):
        return subgraph.Subgraph.Op(self.node)

    def _vars(self, x, *args):
        d = 1.0 / np.sqrt(np.prod(args))
        return {
            'W': self._var(args + (x, ), d),
            'b': self._var(       (x, ), d)
        }
        
    def _var(self, shape, d):
        return tf.Variable(tf.random_uniform(
            shape,
            minval=-d,
            maxval=d
        ))


class Placeholder(subgraph.Subgraph):
    DTYPE = {
        np.int32: tf.int32,
        np.int64: tf.int64,
        np.float32: tf.float32
    }

    def build_graph(self, dtype, shape=None):
        return tf.placeholder(self.DTYPE[dtype], shape=shape)


class Placeholders(subgraph.Subgraph):
    def build_graph(self, variables):
        return Utils.map(
            variables.node,
            lambda v: tf.placeholder(shape=v.get_shape(), dtype=v.dtype)
        )


class Assign(subgraph.Subgraph):
    def build_graph(self, variables, values):
        return [
            tf.assign(variable, value)
            for variable, value in Utils.izip(
                variables.node,
                values.node
            )
        ]

    def assign(self, values):
        return subgraph.Subgraph.Op(self.node, values=values)


class ApplyGradients(subgraph.Subgraph):
    def build_graph(self, weights, gradients, global_step):
        n_steps = np.int64(da3c_config.config.max_global_step)
        reminder = tf.subtract(n_steps, global_step.node['value'])
        factor = tf.cast(reminder, tf.float64) / tf.cast(n_steps, tf.float64)
        learning_rate = tf.maximum(tf.cast(0, tf.float64), factor * da3c_config.config.initial_learning_rate)

        optimizer = tf.train.RMSPropOptimizer(
            learning_rate=learning_rate,
            decay=da3c_config.config.RMSProp.decay,
            momentum=0.0,
            epsilon=da3c_config.config.RMSProp.epsilon
        )
        return tf.group(
            optimizer.apply_gradients(Utils.izip(gradients.node, weights.node)),
            global_step.node['increment']
        )

    def apply_gradients(self, gradients, n_steps):
        return subgraph.Subgraph.Op(self.node, gradients=gradients, n_steps=n_steps)


class Network(subgraph.Subgraph):
    def build_graph(self, state, weights):
        h_conv1 = self._conv(state.node, weights.node['conv1'], 4)
        h_conv2 = self._conv(h_conv1, weights.node['conv2'], 2)

        h_conv2_flat = tf.reshape(h_conv2, [-1, 2592])
        h_fc1 = tf.nn.relu(self._mul(h_conv2_flat, weights.node['fc1']))

        # policy (output)
        pi = tf.nn.softmax(self._mul(h_fc1, weights.node['fc2']))

        # value (output)
        v = tf.reshape(
            tf.matmul(h_fc1, weights.node['fc3']['W']) + weights.node['fc3']['b'],
            [-1]
        )

        return {
            'action': pi,
            'value': v
        }

    def get_action_and_value(self, state):
        return subgraph.Subgraph.Op(self.node, state=state)

    def _conv(self, x, wb, stride):
        return tf.nn.relu(tf.nn.conv2d(x, wb['W'], strides=[1, stride, stride, 1], padding="VALID") + wb['b'])

    def _mul(self, x, wb):
        return tf.matmul(x, wb['W']) + wb['b']


class Loss(subgraph.Subgraph):
    def build_graph(self, state, action, value, discounted_reward, weights, network):
        action_one_hot = tf.one_hot(action.node, da3c_config.config.action_size)

        # avoid NaN with getting the maximum with small value
        log_pi = tf.log(tf.maximum(network.node['action'], 1e-20))

        # policy entropy
        entropy = -tf.reduce_sum(network.node['action'] * log_pi, axis=1)

        # policy loss (output)  (Adding minus, because the original paper's
        # objective function is for gradient ascent, but we use gradient descent optimizer)
        policy_loss = -tf.reduce_sum(
            tf.reduce_sum(log_pi * action_one_hot, axis=1) * (discounted_reward.node - value.node) +
            entropy * da3c_config.config.entropy_beta
        )

        # value loss (output)
        # (Learning rate for Critic is half of Actor's, it's l2 without dividing by 0.5)
        value_loss = tf.reduce_sum(tf.square(discounted_reward.node - network.node['value']))

        # gradient of policy and value are summed up
        loss = policy_loss + value_loss

        self.gradients = Utils.reconstruct(
            tf.gradients(loss, list(Utils.flatten(weights.node))),
            weights.node
        )

    def compute_gradients(self, state, action, value, discounted_reward):
        return subgraph.Subgraph.Op(
            self.gradients,
            state=state,
            action=action,
            value=value,
            discounted_reward=discounted_reward
        )


class LearningRate(subgraph.Subgraph):
    def build_graph(self, n_steps):
        factor = (self._config.max_global_step - global_time_step) / self._config.max_global_step
        learning_rate = self._config.INITIAL_LEARNING_RATE * factor
        if learning_rate < 0.0:
            learning_rate = 0.0
        return learning_rate


class Utils(object):
    @staticmethod
    def map(v, mapping):

        def map_(v):
            if isinstance(v, (tuple, list)):
                return [map_(v1) for v1 in v]
            if isinstance(v, dict):
                return {k: map_(v1) for k, v1 in v.iteritems()}
            return mapping(v)

        return map_(v)

    @staticmethod
    def flatten(v):
        if isinstance(v, (tuple, list)):
            for vv in v:
                for vvv in Utils.flatten(vv):
                    yield vvv
        elif isinstance(v, dict):
            for vv in v.itervalues():
                for vvv in Utils.flatten(vv):
                    yield vvv
        else:
            yield v

    @staticmethod
    def reconstruct(v, pattern):
        i = iter(v)
        result = Utils.map(pattern, lambda v: next(i))
        try:
            next(i)
            assert False
        except StopIteration:
            pass
        return result

    @staticmethod
    def izip(v1, v2):
        if isinstance(v1, (tuple, list)):
            assert isinstance(v2, (tuple, list))
            assert len(v1) == len(v2)
            for vv1, vv2 in itertools.izip(v1, v2):
                for vvv1, vvv2 in Utils.izip(vv1, vv2):
                    yield vvv1, vvv2
        elif isinstance(v1, dict):
            assert isinstance(v2, dict)
            assert len(v1) == len(v2)
            for k1, vv1 in v1.iteritems():
                vv2 = v2[k1]
                for vvv1, vvv2 in Utils.izip(vv1, vv2):
                    yield vvv1, vvv2
        else:
            yield v1, v2
