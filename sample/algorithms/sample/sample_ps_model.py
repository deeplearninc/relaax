import tensorflow as tf
import numpy as np

from relaax.common.algorithms.subgraph import Subgraph


class PSModel(Subgraph):
    def build(self):
        self.step = Variable(0, np.int32)
        self.next_step = Increment(self.step)
        self.state = Placeholder(np.float32, shape=(2, ))
        self.act = Act(self.state, self.next_step)
        self.initialize = Initialize()


class Variable(Subgraph):
    def build(self, initial_value, dtype):
        return tf.Variable(initial_value, dtype)


class Increment(Subgraph):
    def build(self, variable):
        return tf.assign_add(variable.node, 1)


class Placeholder(Subgraph):
    def build(self, dtype, shape):
        return tf.placeholder(np.float32, shape=(2, ))


class Act(Subgraph):
    def build(self, state, next_step):
        with tf.get_default_graph().\
                control_dependencies([next_step.node]):
            return tf.reverse(state.node, [0])


class Initialize(Subgraph):
    def build(self):
        return tf.global_variables_initializer()

