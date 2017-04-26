import itertools
import tensorflow as tf

from relaax.server.common.saver import tensorflow_checkpoint


class Session(object):
    def __init__(self, model):
        self.session = tf.Session()
        self.model = model

    def __getattr__(self, name):
        return SessionMethod(self.session, getattr(self.model, name))

    def make_checkpoint(self):
        return tensorflow_checkpoint.TensorflowCheckpoint(self.session)


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
        op_nodes = [op.node for op in ops]
        return Utils.reconstruct(
            self.session.run(
                list(Utils.flatten(op_nodes)),
                feed_dict=self.flatten_feed_dict(feed_dict)
            ),
            op_nodes
        )

    def flatten_feed_dict(self, feed_dict):
        return {k: v for k, v in self.flatten_fd(feed_dict)}

    def flatten_fd(self, feed_dict):
        for k, v in feed_dict.iteritems():
            for kk, vv in Utils.izip2(k.node, v):
                yield kk, vv


class OpWrapper(object):
    def __init__(self, op):
        self.node = op


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
    def izip2(v1, v2):
        if isinstance(v1, (tuple, list)):
            assert isinstance(v2, (tuple, list))
            assert len(v1) == len(v2)
            for vv1, vv2 in itertools.izip(v1, v2):
                for vvv1, vvv2 in Utils.izip2(vv1, vv2):
                    yield vvv1, vvv2
        elif isinstance(v1, dict):
            assert isinstance(v2, dict)
            assert len(v1) == len(v2)
            for k1, vv1 in v1.iteritems():
                vv2 = v2[k1]
                for vvv1, vvv2 in Utils.izip2(vv1, vv2):
                    yield vvv1, vvv2
        else:
            yield v1, v2
