import tensorflow as tf

from pg_network import SharedParameters

from relaax.common.algorithms.decorators import define_scope, define_input
from relaax.server.parameter_server.parameter_server_base import ParameterServerBase


class PGParameterServer(ParameterServerBase):
    def __init__(self):
        self.graph = PSGraph()
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def run(self, ops, feed_dict):
        return self.on_run(self.session, self.graph, ops, feed_dict)


class PSGraph(SharedParameters):
    def __init__(self):
        # Build TF graph
        super(PSGraph, self).__init__()
        self.step
        self.next_step

    @define_input
    def step(self):
        return tf.Variable(0, tf.int32)

    @define_scope
    def next_step(self):
        return tf.assign_add(self.step, 1)
