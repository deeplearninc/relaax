from __future__ import print_function

import saver


class MultiSaver(saver.Saver):
    def __init__(self, savers):
        super(MultiSaver, self).__init__()
        self._savers = savers

    def global_steps(self):
        steps = set()
        for s in self._savers:
            steps |= s.global_steps()
        return steps

    def remove_checkpoint(self, global_step):
        for s in self._savers:
            s.remove_checkpoint(global_step)

    def restore_checkpoint(self, session, global_step):
        for s in reversed(self._savers):
            if global_step in s.global_steps():
                s.restore_checkpoint(session, global_step)
                break

    def save_checkpoint(self, session, global_step):
        for s in self._savers:
            s.save_checkpoint(session, global_step)
