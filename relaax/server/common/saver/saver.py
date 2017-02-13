from __future__ import print_function


class Saver(object):
    _CHECKPOINT_PREFIX = 'cp'

    def global_steps(self):
        raise NotImplementedError

    def remove_checkpoint(self, global_step):
        raise NotImplementedError

    def restore_checkpoint(self, session, global_step):
        raise NotImplementedError

    def save_checkpoint(self, session, global_step):
        raise NotImplementedError
