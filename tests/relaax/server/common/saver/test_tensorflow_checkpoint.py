import os.path
import ruamel.yaml
import tensorflow
import unittest

from relaax.server.common.saver import tensorflow_checkpoint

class TestTensorflowCheckpoint(unittest.TestCase):
    def setUp(self):
        self._session = tensorflow.Session()
        self._var = tensorflow.Variable(0, tensorflow.int32)
        self._val = tensorflow.placeholder(tensorflow.int32)

        self._session.run(
            tensorflow.variables_initializer(tensorflow.global_variables())
        )
        self._assign = tensorflow.assign(self._var, self._val)

        self.checkpoint = tensorflow_checkpoint.TensorflowCheckpoint(self._session)

    def tearDown(self):
        pass

    def test_checkpoint_ids(self):
        self.assertEquals(set([
            112,
            115
        ]), self.checkpoint.checkpoint_ids([
            'cp-115.ugu',
            'cp-112',
            'cp-115.aga',
            'other_file_name'
        ]))

    def test_checkpoint_names(self, names, checkpoint_id):
        self.assertEquals([
            'cp-115.ugu',
            'cp-115.aga',
        ], self.checkpoint.checkpoint_names([
            'cp-115.ugu',
            'cp-112',
            'cp-115.aga',
            'other_file_name'
        ], 115))

        self.assertEquals([
            'cp-115.ugu',
            'cp-115.aga',
        ], self.checkpoint.checkpoint_names([
            'cp-115.ugu',
            'cp-112',
            'cp-115.aga',
            'other_file_name'
        ], 112))

    def checkpoint_names(self, names, checkpoint_id):

    def test_save_restore(self):
        self._set(14)
        self.checkpoint.save_checkpoint(self.session, 114)

        self._set(13)
        self.checkpoint.save_checkpoint(self.session, 113)

        self._set(2)
        self.checkpoint.save_checkpoint(self.session, 102)

        saver2 = self._saver()

        saver2.restore_checkpoint(self.session, 114)
        self.assertEquals(14, self._get())

        saver2.restore_checkpoint(self.session, 113)
        self.assertEquals(13, self._get())

        saver2.restore_checkpoint(self.session, 102)
        self.assertEquals(2, self._get())

    def test_global_steps(self):
        saver1 = self._saver()
        saver1.save_checkpoint(self.session, 114)
        saver1.save_checkpoint(self.session, 113)
        saver1.save_checkpoint(self.session, 102)
        self.assertEquals(set([102, 113, 114]), saver1.global_steps())

        saver2 = self._saver()
        self.assertEquals(set([102, 113, 114]), saver2.global_steps())

    def test_remove_checkpoint(self):
        saver1 = self._saver()
        saver1.save_checkpoint(self.session, 114)
        saver1.save_checkpoint(self.session, 113)
        saver1.save_checkpoint(self.session, 102)
        saver1.remove_checkpoint(113)
        self.assertEquals(set([102, 114]), saver1.global_steps())

        saver2 = self._saver()
        self.assertEquals(set([102, 114]), saver2.global_steps())

    def _set(self, value):
        self._session.run(self._assign, feed_dict={self._val: value})

    def _get(self):
        return self._session.run(self._var)


if __name__ == '__main__':
    unittest.main()

