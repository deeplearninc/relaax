import tensorflow

from relaax.common.algorithms.decorators import define_scope, define_input
from relaax.server.parameter_server.parameter_server_base import (
    ParameterServerBase)


class SampleParameterServer(ParameterServerBase):
    def __init__(self):
        self.graph = PSGraph()
        self.session = tensorflow.Session()
        self.session.run(tensorflow.global_variables_initializer())

    def run(self, ops, feed_dict):
        return self.on_run(self.session, self.graph, ops, feed_dict)


class PSGraph(object):

    def __init__(self):
        self.step
        self.next_step
        self.state
        self.act

    @define_input
    def step(self):
        return tensorflow.Variable(0, tensorflow.int32)

    @define_scope
    def next_step(self):
        return tensorflow.assign_add(self.step, 1)

    @define_input
    def state(self):
        return tensorflow.placeholder(tensorflow.float32, shape=(2, ))

    @define_scope
    def act(self):
        with tensorflow.get_default_graph().\
                control_dependencies([self.next_step]):
            return tensorflow.reverse(self.state, [0])
