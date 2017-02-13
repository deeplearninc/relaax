import unittest

from relaax.server.common.saver import saver, limited_saver


class TestLimitedSaver(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_global_steps(self):
        saver = limited_saver.LimitedSaver(_MockSaver([11, 200, 14]), None)
        self.assertEquals(set([11, 200, 14]), saver.global_steps())

    def test_remove_checkpoint(self):
        calls = []
        saver = limited_saver.LimitedSaver(
            _MockSaver([11, 200, 14], calls=calls),
            None
        )

        saver.remove_checkpoint(11)
        self.assertEquals([
            ('remove_checkpoint', 11)
        ], calls)

    def test_restore_checkpoint(self):
        calls = []
        saver = limited_saver.LimitedSaver(
            _MockSaver([11, 200, 14], calls=calls),
            None
        )
        session = object()

        saver.restore_checkpoint(session, 199)
        self.assertEquals([
            ('restore_checkpoint', session, 199)
        ], calls)

        del calls[:]
        saver.restore_checkpoint(session, 11)
        self.assertEquals([
            ('restore_checkpoint', session, 11)
        ], calls)

    def test_save_checkpoint_1(self):
        calls = []
        saver = limited_saver.LimitedSaver(
            _MockSaver([11, 200, 14], calls=calls),
            4
        )

        session = object()

        saver.save_checkpoint(session, 12)
        self.assertEquals([
            ('save_checkpoint', session, 12)
        ], calls)

        del calls[:]
        saver.save_checkpoint(session, 19)
        self.assertEquals([
            ('save_checkpoint', session, 19),
            ('remove_checkpoint', 11)
        ], calls)

        del calls[:]
        saver.save_checkpoint(session, 14)
        self.assertEquals([
            ('save_checkpoint', session, 14)
        ], calls)


    def test_save_checkpoint_2(self):
        calls = []
        saver = limited_saver.LimitedSaver(
            _MockSaver([11, 200, 14, 1000, 31], calls=calls),
            2
        )

        session = object()

        saver.save_checkpoint(session, 210)
        self.assertEquals([
            ('save_checkpoint', session, 210),
            ('remove_checkpoint',  11),
            ('remove_checkpoint',  14),
            ('remove_checkpoint',  31),
            ('remove_checkpoint', 200)
        ], calls)


class _MockSaver(saver.Saver):
    def __init__(self, checkpoints, calls=None):
        self._checkpoints = set(checkpoints)
        if calls is None:
            self._calls = []
        else:
            self._calls = calls

    def global_steps(self):
        return self._checkpoints

    def remove_checkpoint(self, global_step):
        self._log('remove_checkpoint', global_step)
        if global_step in self._checkpoints:
            self._checkpoints.remove(global_step)

    def restore_checkpoint(self, session, global_step):
        self._log('restore_checkpoint', session, global_step)

    def save_checkpoint(self, session, global_step):
        self._log('save_checkpoint', session, global_step)
        self._checkpoints.add(global_step)

    def _log(self, *args):
        self._calls.append(args)


if __name__ == '__main__':
    unittest.main()
