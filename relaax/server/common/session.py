import itertools
import tensorflow as tf


class Session(object):
    def __init__(self, model):
        self.session = tf.Session()
        self.model = model

    def __getattr__(self, name):
        return SessionMethod(self.session, getattr(self.model, name))


class SessionMethod(object):
    def __init__(self, session, op):
        self.session = session
        self.op = op

    def __call__(self, **kwargs):
        feed_dict = {
            v: kwargs[k] for k, v in self.op.feed_dict.iteritems()
        }
        result, = self.run(
            [OpWrapper(self.op.subgraph)],
            feed_dict=feed_dict
        )
        return result

    def run(self, ops, feed_dict={}):
        return self.build_result(ops, self.session.run(
            list(self.flatten_ops(ops)),
            feed_dict=self.flatten_feed_dict(feed_dict)
        ))

    def flatten_ops(self, ops):
        for v in self.traverse_values((op.node for op in ops)):
            yield v

    def traverse_values(self, values):
        for value in values:
            for v in self.traverse_value(value):
                yield v

    def traverse_value(self, value):
        if isinstance(value, (tuple, list)):
            for v in self.traverse_values(value):
                yield v
        else:
            yield value

    def flatten_feed_dict(self, feed_dict):
        return {k: v for k, v in self.traverse_pairs(
            ((k.node, v) for k, v in feed_dict.iteritems())
        )}

    def traverse_pairs(self, pairs):
        for pair in pairs:
            for k, v in self.traverse_pair(pair):
                yield k, v

    def traverse_pair(self, pair):
        key, value = pair
        if isinstance(key, (tuple, list)):
            assert isinstance(value, (tuple, list))
            assert len(key) == len(value)
            for k, v in self.traverse_pairs(itertools.izip(key, value)):
                yield k, v
        elif isinstance(key, dict):
            assert isinstance(value, dict)
            for k, v in self.traverse_pairs(self.dict_izip(key, value)):
                yield k, v
        else:
            assert isinstance(key, tf.Tensor)
            yield key, value

    def dict_izip(self, d1, d2):
        assert len(d1) == len(d2)
        for k, v1 in d1.iteritems():
            v2 = d2[k]
            yield v1, v2

    def build_result(self, ops, flat_list):
        i = iter(flat_list)
        result = [self.build_list(op.node, i) for op in ops]
        try:
            next(i)
            assert False
        except StopIteration:
            pass
        return result

    def build_list(self, pattern, i):
        if isinstance(pattern, (tuple, list)):
            return [self.build_list(p, i) for p in pattern]
        return next(i)


class OpWrapper(object):
    def __init__(self, op):
        self.node = op
