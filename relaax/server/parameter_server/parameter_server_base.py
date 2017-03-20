import tensorflow

class ParameterServerBase(object):

    def on_run(self, session, graph, ops, feed_dict):
        fd = {}
        for k, v in feed_dict.iteritems():
            fd[graph.__operations__[k]] = v
        return session.run(
            map(lambda x: graph.__operations__[x], ops),
            feed_dict=fd
        )

class ParameterServerImpl(ParameterServerBase):
    def __init__(self,graph):
        self.graph = graph
        self.session = tensorflow.Session()
        self.session.run(tensorflow.global_variables_initializer())

    def run(self, ops, feed_dict):
        print "ParameterServerImpl run"
        return self.on_run(self.session, self.graph, ops, feed_dict)