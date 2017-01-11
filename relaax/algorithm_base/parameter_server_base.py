

class ParameterServerBase(object):
    def increment_global_t(self):
        raise NotImplementedError

    def apply_gradients(self, gradients):
        raise NotImplementedError

    def get_values(self):
        raise NotImplementedError

    def metrics(self):
        raise NotImplementedError


class ParameterServerBase2(ParameterServerBase):
    def __init__(self, config, saver, metrics):
        raise NotImplementedError

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
