from __future__ import print_function


from builtins import object
class Saver(object):
    def checkpoint_ids(self):
        raise NotImplementedError

    def remove_checkpoint(self, checkpoint_id):
        raise NotImplementedError

    def restore_checkpoint(self, checkpoint_id):
        raise NotImplementedError

    def save_checkpoint(self, checkpoint_id):
        raise NotImplementedError
