from __future__ import print_function

import os
import tensorflow

import saver


class FsSaver(saver.Saver):
    def __init__(self, dir):
        super(FsSaver, self).__init__()
        self._dir = dir

    def restore_latest_checkpoint(self, session):
        return self._restore(self._dir, session)

    def save_checkpoint(self, session, global_step):
        if not os.path.exists(self._dir):
            os.makedirs(self._dir)
        self._save(self._dir, session, global_step)
