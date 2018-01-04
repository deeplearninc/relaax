from __future__ import print_function
from __future__ import absolute_import

from . import decorated_saver


class NormalizedSaver(decorated_saver.DecoratedSaver):
    def __init__(self, saver, checkpoint):
        super(NormalizedSaver, self).__init__(saver)
        self._checkpoint = checkpoint

    def save_checkpoint(self, checkpoint_id):
        super(NormalizedSaver, self).save_checkpoint(self._checkpoint.normalized_checkpoint_id(
            checkpoint_id))
