import contextlib
import shutil
import tempfile
import tensorflow
import unittest

from relaax.server.common.saver import fs_saver
from relaax.server.common.saver import tensorflow_checkpoint

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
            self.assertEquals(set(), self._saver(dir).checkpoint_ids())

    def test_save_restore(self):
        with _temp_dir() as dir:
            saver1 = self._saver(dir)

            self._set(14)
            saver1.save_checkpoint(114)

            self._set(13)
            saver1.save_checkpoint(113)

            self._set(2)
            saver1.save_checkpoint(102)

            saver2 = self._saver(dir)

            saver2.restore_checkpoint(114)
            self.assertEquals(14, self._get())

            saver2.restore_checkpoint(113)
            self.assertEquals(13, self._get())

            saver2.restore_checkpoint(102)
            self.assertEquals(2, self._get())

    def test_global_steps(self):
        with _temp_dir() as dir:
            saver1 = self._saver(dir)
            saver1.save_checkpoint(114)
            saver1.save_checkpoint(113)
            saver1.save_checkpoint(102)
            self.assertEquals(set([102, 113, 114]), saver1.checkpoint_ids())

            saver2 = self._saver(dir)
            self.assertEquals(set([102, 113, 114]), saver2.checkpoint_ids())

    def test_remove_checkpoint(self):
        with _temp_dir() as dir:
            saver1 = self._saver(dir)
            saver1.save_checkpoint(114)
            saver1.save_checkpoint(113)
            saver1.save_checkpoint(102)
            saver1.remove_checkpoint(113)
            self.assertEquals(set([102, 114]), saver1.checkpoint_ids())

            saver2 = self._saver(dir)
            self.assertEquals(set([102, 114]), saver2.checkpoint_ids())

    def _set(self, value):
        self.session.run(self._assign, feed_dict={self._val: value})

    def _get(self):
        return self.session.run(self._var)

    def _saver(self, dir):
        return fs_saver.FsSaver(
            checkpoint=tensorflow_checkpoint.TensorflowCheckpoint(self.session),
            dir=dir
        )


@contextlib.contextmanager
def _temp_dir():
    path = tempfile.mkdtemp()
    try:
        yield path
    finally:
        shutil.rmtree(path)


if __name__ == '__main__':
    unittest.main()
