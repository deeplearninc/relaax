from __future__ import print_function
from __future__ import absolute_import

from . import saver


class DecoratedSaver(saver.Saver):
    def __init__(self, saver):
        super(DecoratedSaver, self).__init__()
        self._saver = saver

    def checkpoint_ids(self):
        return self._saver.checkpoint_ids()

    def remove_checkpoint(self, checkpoint_id):
        self._saver.remove_checkpoint(checkpoint_id)

    def restore_checkpoint(self, checkpoint_id):
        self._saver.restore_checkpoint(checkpoint_id)

    def save_checkpoint(self, checkpoint_id):
        self._saver.save_checkpoint(checkpoint_id)
