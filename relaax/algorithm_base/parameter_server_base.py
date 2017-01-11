

class ParameterServerBase2(object):
    def __init__(self, config, saver, metrics):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def restore_latest_checkpoint(self):
        raise NotImplementedError

    def save_checkpoint(self):
        raise NotImplementedError

    def checkpoint_location(self):
        raise NotImplementedError

    def global_t(self):
        raise NotImplementedError
