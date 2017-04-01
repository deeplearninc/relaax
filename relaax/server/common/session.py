import tensorflow as tf


class Session(object):
    def __init__(self, graph):
        self.session = tf.Session()
        self.graph = graph

    def run(self, ops, feed_dict={}):
        return self.build_list(ops, self.session.run(
            list(self.flatten_list(ops)),
            feed_dict=self.flatten_dict(feed_dict)
        ))

    def flatten_list(self, values):
        for value in values:
            for vv in self.flatten_l(value.node):
                yield vv

    def flatten_l(self, v):
        if isinstance(v, (tuple, list)):
            for vv in v:
                for vvv in self.flatten_l(vv):
                    yield vv
        else:
            yield v

    def flatten_dict(self, feed_dict):
        return {k: v for k, v in self.flatten_d(feed_dict)}

    def flatten_d(self, feed_dict):
        for key, v in feed_dict.iteritems():
            k = key.node
            if isinstance(k, (tuple, list)):
                assert isinstance(v, (tuple, list))
                assert len(k) == len(v)
                for kk, vv in zip(k, v):
                    yield kk, vv
            else:
                yield k, v

    def build_list(self, ops, flat_list):
        i = iter(flat_list)
        return [self.build_l(op.node, i) for op in ops]
    
    def build_l(self, pattern, i):
        if isinstance(pattern, (tuple, list)):
            return [self.build_l(p, i) for p in pattern]
        return next(i)


