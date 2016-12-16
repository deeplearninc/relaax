from __future__ import print_function

import tensorflow


class Saver(object):
    _CHECKPOINT_PREFIX = 'cp'
    _LATEST_FILENAME = 'latest'

    def __init__(self):
        self._saver = None

    def _restore(self, dir, session):
        cp_path = self._latest_cp_path(dir)
        if cp_path is not None:
            self._get_saver().restore(session, cp_path)
            return True
        return False

    def _save(self, dir, session, global_step):
        self._get_saver().save(
            session,
            '%s/%s' % (dir, self._CHECKPOINT_PREFIX),
            global_step=global_step,
            latest_filename=self._LATEST_FILENAME
        )

    def _latest_cp_path(self, dir):
        return tensorflow.train.latest_checkpoint(dir, latest_filename=self._LATEST_FILENAME)

    def _get_saver(self):
        if self._saver is None:
            self._saver = tensorflow.train.Saver()
        return self._saver
