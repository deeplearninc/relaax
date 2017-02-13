from __future__ import print_function

import saver


class LimitedSaver(saver.Saver):
    def __init__(self, saver, limit):
        super(LimitedSaver, self).__init__()
        self._saver = saver
        self._limit = limit

    def global_steps(self):
        return self._saver.global_steps()

    def remove_checkpoint(self, global_step):
        self._saver.remove_checkpoint(global_step)

    def restore_checkpoint(self, session, global_step):
        self._saver.restore_checkpoint(session, global_step)

    def save_checkpoint(self, session, global_step):
        self._saver.save_checkpoint(session, global_step)
        global_steps = self._saver.global_steps()
        if len(global_steps) > self._limit:
            for global_step in sorted(global_steps)[:-self._limit]:
                self._saver.remove_checkpoint(global_step)
