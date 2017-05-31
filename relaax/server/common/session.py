from builtins import next
from builtins import object
import itertools
import tensorflow as tf

from relaax.server.common.saver import tensorflow_checkpoint


class Session(object):
    def __init__(self, *args, **kwargs):
        self.session = tf.Session()
        if len(args) == 0:
            assert len(kwargs) > 0
            self.model = SuperModel(kwargs)
        else:
            assert len(kwargs) == 0
            self.model, = args


    def __getattr__(self, name):
        # print('name', name)
        return SessionMethod(self.session, getattr(self.model, name))

    def make_checkpoint(self):
        return tensorflow_checkpoint.TensorflowCheckpoint(self.session)


class SuperModel(object):
    def __init__(self, desc):
        for k, v in desc.items():
            if isinstance(v, dict):
                submodel = SuperModel(v)
            else:
                submodel = v
            setattr(self, k, submodel)


class SessionMethod(object):
    def __init__(self, session, op_or_model):
        self.session = session
        self.op_or_model = op_or_model

    def __getattr__(self, name):
        # print('name', name)
        return SessionMethod(self.session, getattr(self.op_or_model, name))

    def __call__(self, *args, **kwargs):
        return self.op_or_model(self.session, *args, **kwargs)
