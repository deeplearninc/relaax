from __future__ import print_function

import os
import re
import tensorflow

import checkpoint


class TensorflowCheckpoint(checkpoint.Checkpoint):
    _CHECKPOINT_PREFIX = 'cp'

    def __init__(self, session):
        self._session = session
        self.saver = tensorflow.train.Saver()

    def checkpoint_ids(self, names):
        re_ = re.compile('^%s-(\d+)(|\..+)$' % self._CHECKPOINT_PREFIX)
        ids = set()
        for name in names:
            match = re_.match(name)
            if match is not None:
                ids.add(int(match.group(1)))
        return ids

    def checkpoint_names(self, names, checkpoint_id):
        re_ = re.compile('^%s-%d(?:|\..+)$' % (self._CHECKPOINT_PREFIX, checkpoint_id))
        return (name for name in names if re_.match(name) is not None)

    def restore_checkpoint(self, dir, checkpoint_id):
        self.saver.restore(
            self._session,
            os.path.join(dir, '%s-%d' % (self._CHECKPOINT_PREFIX, checkpoint_id))
        )

    def save_checkpoint(self, dir, checkpoint_id):
        self.saver.save(
            self._session,
            os.path.join(dir, self._CHECKPOINT_PREFIX),
            global_step=checkpoint_id
        )
