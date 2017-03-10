from __future__ import print_function

import saver


class MultiSaver(saver.Saver):
    def __init__(self, savers):
        super(MultiSaver, self).__init__()
        self._savers = savers

    def checkpoint_ids(self):
        steps = set()
        for s in self._savers:
            steps |= s.checkpoint_ids()
        return steps

    def remove_checkpoint(self, checkpoint_id):
        for s in self._savers:
            s.remove_checkpoint(checkpoint_id)

    def restore_checkpoint(self, checkpoint_id):
        for s in reversed(self._savers):
            if checkpoint_id in s.checkpoint_ids():
                s.restore_checkpoint(checkpoint_id)
                break

    def save_checkpoint(self, checkpoint_id):
        for s in self._savers:
            s.save_checkpoint(checkpoint_id)
