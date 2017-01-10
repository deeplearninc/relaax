

class ParameterServerBase(object):
    def increment_global_t(self):
        raise NotImplementedError

    def apply_gradients(self, gradients):
        raise NotImplementedError

    def get_values(self):
        raise NotImplementedError

    def store_scalar_metric(self, name, y, x=None):
        raise NotImplementedError
