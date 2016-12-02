from __future__ import print_function

import tensorflow


class Saver(object):
    _CHECKPOINT_PREFIX = 'cp'
    _LATEST_FILENAME = 'latest'

    def __init__(self):
        self._saver = tensorflow.train.Saver()

    def _restore(self, dir, session):
        cp_path = self._latest_cp_path(self._dir, latest_filename=_LATEST_FILENAME)
        if cp_path is not None:
            self._saver.restore(session, cp_path)
            return True
        return False

    def _save(self, dir, session, global_step):
        self._saver.save(
            session,
            '%s/%s' % (self._dir, _CHECKPOINT_PREFIX),
            global_step=global_step,
            latest_filename=_LATEST_FILENAME
        )

    def _latest_cp_path(self, dir):
        return tensorflow.train.latest_checkpoint(dir, latest_filename=_LATEST_FILENAME)
