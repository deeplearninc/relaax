from __future__ import print_function
from __future__ import absolute_import

import logging
import os
import re
import tensorflow

from . import saver


_logger = logging.getLogger(__name__)


class FsSaver(saver.Saver):
    def __init__(self, checkpoint, dir):
        super(FsSaver, self).__init__()
        self._checkpoint = checkpoint
        self._dir = dir

    def checkpoint_ids(self):
        names = []
        if os.path.exists(self._dir):
            names = os.listdir(self._dir)
        return self._checkpoint.checkpoint_ids(names)

    def remove_checkpoint(self, checkpoint_id):
        removed = False
        for name in self._checkpoint.checkpoint_names(os.listdir(self._dir), checkpoint_id):
            os.remove(os.path.join(self._dir, name))
            removed = True
        if removed:
            _logger.info('checkpoint {} was removed from {} dir'.format(checkpoint_id, self._dir))

    def restore_checkpoint(self, checkpoint_id):
        self._checkpoint.restore_checkpoint(self._dir, checkpoint_id)
        _logger.info('checkpoint {} was restored from {} dir'.format(checkpoint_id, self._dir))

    def save_checkpoint(self, checkpoint_id):
        if not os.path.exists(self._dir):
            os.makedirs(self._dir)
        self._checkpoint.save_checkpoint(self._dir, checkpoint_id)
        _logger.info('checkpoint {} was saved to {} dir'.format(checkpoint_id, self._dir))
