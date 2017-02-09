import contextlib
import shutil
import tempfile
import tensorflow
import unittest

from relaax.server.common.saver import fs_saver

class TestFsSaver(unittest.TestCase):
    def setUp(self):
        self.session = tensorflow.Session()
        self._var = tensorflow.Variable(0, tensorflow.int32)
        self._val = tensorflow.placeholder(tensorflow.int32)

        self.session.run(
            tensorflow.variables_initializer(tensorflow.global_variables())
        )
        self._assign = tensorflow.assign(self._var, self._val)

    def tearDown(self):
        pass

    def test_no_checkpoints(self):
        with _temp_dir() as dir:
            saver = fs_saver.FsSaver(dir)
            self.assertEquals(set(), saver.global_steps())

    def test_save_restore(self):
        with _temp_dir() as dir:
            saver1 = fs_saver.FsSaver(dir)

            self._set(14)
            saver1.save_checkpoint(self.session, 114)

            self._set(13)
            saver1.save_checkpoint(self.session, 113)

            self._set(2)
            saver1.save_checkpoint(self.session, 102)

            saver2 = fs_saver.FsSaver(dir)

            saver2.restore_checkpoint(self.session, 114)
            self.assertEquals(14, self._get())

            saver2.restore_checkpoint(self.session, 113)
            self.assertEquals(13, self._get())

            saver2.restore_checkpoint(self.session, 102)
            self.assertEquals(2, self._get())

    def test_global_steps(self):
        with _temp_dir() as dir:
            saver1 = fs_saver.FsSaver(dir)
            saver1.save_checkpoint(self.session, 114)
            saver1.save_checkpoint(self.session, 113)
            saver1.save_checkpoint(self.session, 102)
            self.assertEquals(set([102, 113, 114]), saver1.global_steps())

            saver2 = fs_saver.FsSaver(dir)
            self.assertEquals(set([102, 113, 114]), saver2.global_steps())

    def test_remove_checkpoint(self):
        with _temp_dir() as dir:
            saver1 = fs_saver.FsSaver(dir)
            saver1.save_checkpoint(self.session, 114)
            saver1.save_checkpoint(self.session, 113)
            saver1.save_checkpoint(self.session, 102)
            saver1.remove_checkpoint(113)
            self.assertEquals(set([102, 114]), saver1.global_steps())

            saver2 = fs_saver.FsSaver(dir)
            self.assertEquals(set([102, 114]), saver2.global_steps())

    def _set(self, value):
        self.session.run(self._assign, feed_dict={self._val: value})

    def _get(self):
        return self.session.run(self._var)


@contextlib.contextmanager
def _temp_dir():
    path = tempfile.mkdtemp()
    try:
        yield path
    finally:
        shutil.rmtree(path)
