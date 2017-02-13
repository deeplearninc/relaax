from __future__ import print_function

import logging
import os
import re
import tensorflow

import saver


_logger = logging.getLogger(__name__)


class FsSaver(saver.Saver):
    def __init__(self, dir):
        super(FsSaver, self).__init__()
        self._dir = dir

    def global_steps(self):
        steps = set()
        if os.path.exists(self._dir):
            for name in os.listdir(self._dir):
                match = re.match('^%s-(\d+)(|\..+)$' % self._CHECKPOINT_PREFIX, name)
                if match is not None:
                    steps.add(int(match.group(1)))
        return steps

    def remove_checkpoint(self, global_step):
        removed = False
        for name in os.listdir(self._dir):
            match = re.match('^%s-%d(?:|\..+)$' % (self._CHECKPOINT_PREFIX, global_step), name)
            if match is not None:
                os.remove(os.path.join(self._dir, name))
                removed = True
        if removed:
            _logger.info('checkpoint {} was removed from {} dir'.format(global_step, self._dir))

    def restore_checkpoint(self, session, global_step):
        tensorflow.train.Saver().restore(
            session,
            os.path.join(self._dir, '%s-%d' % (self._CHECKPOINT_PREFIX, global_step))
        )
        _logger.info('checkpoint {} was restored from {} dir'.format(global_step, self._dir))

    def save_checkpoint(self, session, global_step):
        if not os.path.exists(self._dir):
            os.makedirs(self._dir)
        tensorflow.train.Saver().save(
            session,
            '%s/%s' % (self._dir, self._CHECKPOINT_PREFIX),
            global_step=global_step
        )
        _logger.info('checkpoint {} was saved to {} dir'.format(global_step, self._dir))
