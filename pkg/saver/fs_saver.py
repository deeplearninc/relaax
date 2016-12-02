from __future__ import print_function

import tensorflow

import saver


class FsSaver(Saver):
    def __init__(self, dir):
        super(FsSaver, self).__init__()
        self._dir = dir

    def restore_latest_checkpoint(self, session):
        return self._restore(session, self._dir)

    def save_checkpoint(self, session, global_step):
        if not os.path.exists(self._dir):
            os.makedirs(self._dir)
        self._save(self, self._dir, session, global_step)
