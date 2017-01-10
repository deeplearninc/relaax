

class ParameterServerBase(object):
    def close(self):
        raise NotImplementedError

    def restore_latest_checkpoint(self):
        raise NotImplementedError

    def save_checkpoint(self):
        raise NotImplementedError

    def checkpoint_place(self):
        raise NotImplementedError

    def global_t(self):
        raise NotImplementedError

    def increment_global_t(self):
        raise NotImplementedError

    def apply_gradients(self, gradients):
        raise NotImplementedError

    def get_values(self):
        raise NotImplementedError

    def store_scalar_metric(self, name, y, x=None):
        raise NotImplementedError

    def service(self):
        raise NotImplementedError
