from builtins import object
class ParameterServerBase(object):
    def __init__(self, saver_factory, metrics_factory):
        self.init_session()
        assert hasattr(self, 'session')
        self.saver = saver_factory(self.create_checkpoint())
        self.metrics = metrics_factory(self.n_step)

    def close(self):
        self.session.close()

    def init_session(self):
        raise NotImplementedError

    def restore_latest_checkpoint(self):
        checkpoint_ids = self.saver.checkpoint_ids()
        if len(checkpoint_ids) > 0:
            self.saver.restore_checkpoint(max(checkpoint_ids))

    def save_checkpoint(self):
        self.saver.save_checkpoint(self.n_step())

    def create_checkpoint(self):
        return self.session.create_checkpoint()

    def n_step(self):
        raise NotImplementedError
