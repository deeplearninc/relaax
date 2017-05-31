from builtins import object
import tensorflow as tf


class Subgraph(object):

    def __init__(self, *args, **kwargs):
        with tf.variable_scope(type(self).__name__):
            self.__node = self.build_graph(*args, **kwargs)

    @property
    def node(self):
        return self.__node

    def Op(self, op, **feed_dict):
        return Ops([op], feed_dict)

    def Ops(self, *ops, **feed_dict):
        return Ops(ops, feed_dict)

    def Call(self, f):
        return Call(f)


class Call(object):
    def __init__(self, f):
        self.f = f

    def __call__(self, session, *args, **kwargs):
        return self.f(session, *args, **kwargs)


class Ops(object):
    def __init__(self, ops, feed_dict):
        self.ops = ops
        self.feed_dict = feed_dict

    def __call__(self, session, **kwargs):
        ops = [self.cast(op) for op in self.ops]
        feed_dict = {v: kwargs[k] for k, v in self.feed_dict.items()}
        # print('feed_dict')
        # for k, v in self.flatten_feed_dict(feed_dict).items():
        #     import numpy as np
        #     print(repr(k), repr(np.asarray(v).shape))
        result = Utils.reconstruct(
            session.run(
                list(Utils.flatten(ops)),
                feed_dict=self.flatten_feed_dict(feed_dict)
            ),
            ops
        )
        if len(ops) == 1:
            result, = result
        return result

    def flatten_feed_dict(self, feed_dict):
        return {k: v for k, v in self.flatten_fd(feed_dict)}

    def flatten_fd(self, feed_dict):
        for k, v in feed_dict.items():
            for kk, vv in Utils.izip2(self.cast(k), v):
                yield kk, vv

    def cast(self, v):
        if isinstance(v, Subgraph):
            return v.node
        return v


class Utils(object):
    @staticmethod
    def map(v, mapping):

        def map_(v):
            if isinstance(v, (tuple, list)):
                return [map_(v1) for v1 in v]
            if isinstance(v, dict):
                return {k: map_(v1) for k, v1 in v.items()}
            return mapping(v)

        return map_(v)

    @classmethod
    def flatten(cls, v):
        if isinstance(v, (tuple, list)):
            for vv in v:
                for vvv in cls.flatten(vv):
                    yield vvv
        elif isinstance(v, dict):
            for vv in v.values():
                for vvv in cls.flatten(vv):
                    yield vvv
        else:
            yield v

    @classmethod
    def reconstruct(cls, v, pattern):
        i = iter(v)
        result = cls.map(pattern, lambda v: next(i))
        try:
            next(i)
            assert False
        except StopIteration:
            pass
        return result

    @classmethod
    def izip2(cls, v1, v2):
        if isinstance(v1, (tuple, list)):
            assert isinstance(v2, (tuple, list))
            assert len(v1) == len(v2)
            for vv1, vv2 in zip(v1, v2):
                for vvv1, vvv2 in cls.izip2(vv1, vv2):
                    yield vvv1, vvv2
        elif isinstance(v1, dict):
            assert isinstance(v2, dict)
            assert len(v1) == len(v2)
            for k1, vv1 in v1.items():
                vv2 = v2[k1]
                for vvv1, vvv2 in cls.izip2(vv1, vv2):
                    yield vvv1, vvv2
        else:
            yield v1, v2
