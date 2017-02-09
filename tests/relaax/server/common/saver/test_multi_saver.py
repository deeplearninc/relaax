import unittest

from relaax.server.common.saver import saver, multi_saver


class TestMultiSaver(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_global_steps(self):
        saver = multi_saver.MultiSaver([
            _MockSaver([10, 200, 15]),
            _MockSaver([11, 200, 14])
        ])
        self.assertEquals(set([10, 200, 11, 14, 15]), saver.global_steps())

    def test_remove_checkpoint(self):
        calls = []
        saver = multi_saver.MultiSaver([
            _MockSaver([10, 200, 15], id=0, calls=calls),
            _MockSaver([11, 200, 14], id=1, calls=calls)
        ])

        saver.remove_checkpoint(10)
        self.assertEquals([
            (0, 'remove_checkpoint', 10),
            (1, 'remove_checkpoint', 10)
        ], calls)

    def test_restore_checkpoint(self):
        calls = []
        saver = multi_saver.MultiSaver([
            _MockSaver([10, 200, 15], id=0, calls=calls),
            _MockSaver([11, 200, 14], id=1, calls=calls)
        ])
        session = object()

        saver.restore_checkpoint(session, 199)
        self.assertEquals([
            (1, 'global_steps'),
            (0, 'global_steps')
        ], calls)

        del calls[:]
        saver.restore_checkpoint(session, 11)
        self.assertEquals([
            (1, 'global_steps'),
            (1, 'restore_checkpoint', session, 11)
        ], calls)

        del calls[:]
        saver.restore_checkpoint(session, 10)
        self.assertEquals([
            (1, 'global_steps'),
            (0, 'global_steps'),
            (0, 'restore_checkpoint', session, 10)
        ], calls)

        del calls[:]
        saver.restore_checkpoint(session, 200)
        self.assertEquals([
            (1, 'global_steps'),
            (1, 'restore_checkpoint', session, 200)
        ], calls)

    def test_save_checkpoint(self):
        calls = []
        saver = multi_saver.MultiSaver([
            _MockSaver([], id=0, calls=calls),
            _MockSaver([], id=1, calls=calls)
        ])

        session = object()

        saver.save_checkpoint(session, 10)
        self.assertEquals([
            (0, 'save_checkpoint', session, 10),
            (1, 'save_checkpoint', session, 10)
        ], calls)


class _MockSaver(saver.Saver):
    def __init__(self, checkpoints, id=None, calls=None):
        self._checkpoints = set(checkpoints)
        self._id = id
        if calls is None:
            self._calls = []
        else:
            self._calls = calls

    def global_steps(self):
        self._log('global_steps')
        return self._checkpoints

    def remove_checkpoint(self, global_step):
        self._log('remove_checkpoint', global_step)
        if global_step in self._checkpoints:
            self._checkpoints.remove(global_step)

    def restore_checkpoint(self, session, global_step):
        self._log('restore_checkpoint', session, global_step)

    def save_checkpoint(self, session, global_step):
        self._log('save_checkpoint', session, global_step)

    def _log(self, *args):
        self._calls.append((self._id, ) + args)



if __name__ == '__main__':
    unittest.main()
