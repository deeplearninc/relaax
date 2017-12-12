from __future__ import absolute_import

import os
import re
import tensorflow

from . import checkpoint


class TensorflowCheckpoint(checkpoint.Checkpoint):
    _CHECKPOINT_PREFIX = 'cp'
    _CHECKPOINT_RE_PREFIX = _CHECKPOINT_PREFIX

    def __init__(self, session):
        self.session = session
        self.saver = tensorflow.train.Saver(max_to_keep=None)

    def checkpoint_ids(self, names):
        re_ = re.compile('^%s-(\d+)(|\..+)$' % self._CHECKPOINT_RE_PREFIX)
        ids = set()
        for name in names:
            match = re_.match(name)
            if match is not None:
                ids.add(self._parse_match(match))
        return ids

    def checkpoint_names(self, names, checkpoint_id):
        prefix = self._full_prefix(checkpoint_id)
        re_ = re.compile('^%s(?:|\..+)$' % re.escape(prefix))
        return (name for name in names if re_.match(name) is not None)

    def restore_checkpoint(self, dir, checkpoint_id):
        prefix = self._full_prefix(checkpoint_id)
        self.saver.restore(self.session, os.path.join(dir, prefix))

    def save_checkpoint(self, dir, checkpoint_id):
        self.saver.save(self.session, os.path.join(dir, self._short_prefix(checkpoint_id)),
                        global_step=self._n_step(checkpoint_id))

    def _parse_match(self, match):
        n_step = int(match.group(1))
        return n_step

    def _short_prefix(self, checkpoint_id):
        return self._CHECKPOINT_PREFIX

    def _n_step(self, checkpoint_id):
        return checkpoint_id

    def _full_prefix(self, checkpoint_id):
        return '%s-%d' % (self._short_prefix(checkpoint_id), self._n_step(checkpoint_id))


class TensorflowScoredCheckpoint(TensorflowCheckpoint):
    _CHECKPOINT_PREFIX = 'best-cp'
    _CHECKPOINT_RE_PREFIX = '%s-(\d+(?:\.\d+))' % _CHECKPOINT_PREFIX

    def _parse_match(self, match):
        score = float(match.group(1))
        n_step = int(match.group(2))
        return n_step, score

    def _short_prefix(self, checkpoint_id):
        _, score = checkpoint_id

        score_s = '%f' % score

        # trim trailing zeros after decimal point
        score_s = re.sub('(\.\d*[1-9])0+$', '\\1', score_s)

        # remove useless point and trailing zeros
        score_s = re.sub('\.0+$', '', score_s)

        return '%s-%s' % (self._CHECKPOINT_PREFIX, score_s)

    def _n_step(self, checkpoint_id):
        n_step, _ = checkpoint_id
        return n_step
