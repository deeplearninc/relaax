from builtins import object
import tensorflow as tf
import re


class Subgraph(object):

    def __init__(self, *args, **kwargs):
        with tf.variable_scope(None, default_name=type(self).__name__):
            self.__node = self.build_graph(*args, **kwargs)

    @property
    def node(self):
        return self.__node

    def Op(self, op, **feed_dict):
        return Op(op, feed_dict)

    def Ops(self, *ops, **feed_dict):
        return Op(ops, feed_dict)

    def Call(self, f):
        return Call(f)


class Call(object):
    def __init__(self, f):
        self.f = f

    def __call__(self, session, *args, **kwargs):
        return self.f(session, *args, **kwargs)


class Op(object):
    def __init__(self, op, feed_dict):
        self.op = op
        self.feed_dict = feed_dict

    def __call__(self, session, **kwargs):
        feed_dict = {v: kwargs[k] for k, v in self.feed_dict.items()}
        # print('feed_dict')
        # for k, v in self.flatten_feed_dict(feed_dict).items():
        #     import numpy as np
        #     print(repr(k), repr(np.asarray(v).shape))
        return self.reconstruct(session._tf_session.run(list(self.flatten(self.op)),
                                feed_dict=self.flatten_feed_dict(feed_dict)), self.op)

    @classmethod
    def flatten_feed_dict(cls, feed_dict):
        return {k: v for k, v in cls.flatten_fd(feed_dict)}

    @classmethod
    def flatten_fd(cls, feed_dict):
        for k, v in feed_dict.items():
            for kk, vv in cls.izip2(k, v):
                yield kk, vv

    @classmethod
    def map(cls, v, mapping):

        def _map(v):
            v = cls.cast(v)
            if isinstance(v, (tuple, list)):
                return [_map(v1) for v1 in v]
            if isinstance(v, dict):
                return {k: _map(v1) for k, v1 in v.items()}
            return mapping(v)

        return _map(v)

    @classmethod
    def flatten(cls, v):
        v = cls.cast(v)
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
        v1 = cls.cast(v1)
        if isinstance(v1, (tuple, list)):
            assert isinstance(v2, (tuple, list))
            #print("v1 = {}, v2 = {}".format(len(v1), len(v2)))
            #if len(v1) != len(v2):
            #    print("v1 = {}, v2 = {}".format(v1, v2))
            #    print("v1 = {}, v2 = {}".format(only_brackets(repr(v1)), only_brackets(repr(v2))))
            assert len(v1) == len(v2)
            for vv1, vv2 in zip(v1, v2):
                for vvv1, vvv2 in cls.izip2(vv1, vv2):
                    yield vvv1, vvv2
        elif isinstance(v1, dict):
            assert isinstance(v2, dict)
            #print("v1 = {}, v2 = {}".format(len(v1), len(v2)))
            #if len(v1) != len(v2):
            #    print("v1 = {}, v2 = {}".format(v1, v2))
            #    print("v1 = {}, v2 = {}".format(only_brackets(repr(v1)), only_brackets(repr(v2))))
            assert len(v1) == len(v2)
            for k1, vv1 in v1.items():
                vv2 = v2[k1]
                for vvv1, vvv2 in cls.izip2(vv1, vv2):
                    yield vvv1, vvv2
        else:
            yield v1, v2

    @staticmethod
    def cast(v):
        if isinstance(v, Subgraph):
            return v.node
        return v


def only_brackets(s):
    s1 = re.sub("[^\[\]]+", "", s)
    s2 = s1.replace("][", "], [")
    return s2