import tensorflow as tf
import numpy as np

from relaax.common.algorithms.subgraph import Subgraph


class PSModel(Subgraph):
    def build_graph(self):
        sg_step = Variable(0, np.int32)
        sg_next_step = Increment(sg_step)
        ph_state = Placeholder(np.float32, shape=(2, ))
        sg_act = Act(ph_state, sg_next_step)

        self.op_step = sg_step.value()
        self.op_act = sg_act.act(ph_state)
        self.op_initialize = Initialize().initialize()


class Variable(Subgraph):
    def build_graph(self, initial_value, dtype):
        return tf.Variable(initial_value, dtype)

    def value(self):
        return Subgraph.Op(self.node)


class Increment(Subgraph):
    def build_graph(self, variable):
        return tf.assign_add(variable.node, 1)


class Placeholder(Subgraph):
    def build_graph(self, dtype, shape):
        return tf.placeholder(np.float32, shape=(2, ))


class Act(Subgraph):
    def build_graph(self, state, next_step):
        with tf.get_default_graph().\
                control_dependencies([next_step.node]):
            return tf.reverse(state.node, [0])

    def act(self, state):
        return Subgraph.Op(self.node, state=state)


class Initialize(Subgraph):
    def build_graph(self):
        return tf.global_variables_initializer()

    def initialize(self):
        return Subgraph.Op(self.node)
