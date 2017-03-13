
class Checkpoint(object):
    def checkpoint_ids(self, names):
        raise NotImplementedError

    def checkpoint_names(self, names, checkpoint_id):
        raise NotImplementedError

    def restore_checkpoint(self, dir, checkpoint_id):
        raise NotImplementedError

    def save_checkpoint(self, dir, checkpoint_id):
        raise NotImplementedError
