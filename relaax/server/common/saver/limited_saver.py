from __future__ import print_function

import saver


class LimitedSaver(saver.Saver):
    def __init__(self, saver, limit):
        super(LimitedSaver, self).__init__()
        self._saver = saver
        self._limit = limit

    def checkpoint_ids(self):
        return self._saver.checkpoint_ids()

    def remove_checkpoint(self, checkpoint_id):
        self._saver.remove_checkpoint(checkpoint_id)

    def restore_checkpoint(self, checkpoint_id):
        self._saver.restore_checkpoint(checkpoint_id)

    def save_checkpoint(self, checkpoint_id):
        self._saver.save_checkpoint(checkpoint_id)
        checkpoint_ids = self._saver.checkpoint_ids()
        if len(checkpoint_ids) > self._limit:
            for checkpoint_id in sorted(checkpoint_ids)[:-self._limit]:
                self._saver.remove_checkpoint(checkpoint_id)
