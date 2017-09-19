from builtins import next
from builtins import object
import itertools
import tensorflow as tf

from relaax.server.common.saver import tensorflow_checkpoint


class Session(object):
    def __init__(self, *args, **kwargs):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self._parent_session = None
        self._name = 'root_session'
        self._tf_session = tf.Session(config=config)
        if len(args) == 0:
            assert len(kwargs) > 0
            self.model = SuperModel(kwargs)
        else:
            assert len(kwargs) == 0
            self.model, = args


    def __getattr__(self, name):
        return SessionMethod(self, name, getattr(self.model, name))

    def create_checkpoint(self):
        return tensorflow_checkpoint.TensorflowCheckpoint(self._tf_session)


class SuperModel(object):
    def __init__(self, desc):
        for k, v in desc.items():
            if isinstance(v, dict):
                submodel = SuperModel(v)
            else:
                submodel = v
            setattr(self, k, submodel)


class SessionMethod(object):
    def __init__(self, parent_session, name, op_or_model):
        self._parent_session = parent_session
        self._name = name
        self._tf_session = parent_session._tf_session
        self._op_or_model = op_or_model

    def __getattr__(self, name):
        return SessionMethod(self, name, getattr(self._op_or_model, name))

    def __call__(self, *args, **kwargs):
        return self._op_or_model(self._parent_session, *args, **kwargs)

    def _full_path(self):
        s = self
        r = ''
        while s is not None:
            if r != '':
                r = '.' + r
            r = s._name + r
            s = s._parent_session
        return r
