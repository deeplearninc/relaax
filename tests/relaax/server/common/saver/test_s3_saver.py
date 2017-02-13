import os.path
import ruamel.yaml
import tensorflow
import unittest

from relaax.server.common.saver import s3_saver

class TestS3Saver(unittest.TestCase):
    def setUp(self):
        self.session = tensorflow.Session()
        self._var = tensorflow.Variable(0, tensorflow.int32)
        self._val = tensorflow.placeholder(tensorflow.int32)

        self.session.run(
            tensorflow.variables_initializer(tensorflow.global_variables())
        )
        self._assign = tensorflow.assign(self._var, self._val)

        self._clean()

    def tearDown(self):
        self._clean()

    def test_no_checkpoints(self):
        saver = self._saver()
        self.assertEquals(set(), saver.global_steps())

    def test_save_restore(self):
        saver1 = self._saver()

        self._set(14)
        saver1.save_checkpoint(self.session, 114)

        self._set(13)
        saver1.save_checkpoint(self.session, 113)

        self._set(2)
        saver1.save_checkpoint(self.session, 102)

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
        self.session.run(self._assign, feed_dict={self._val: value})

    def _get(self):
        return self.session.run(self._var)

    def _saver(self):
        keys = _load_yaml(
            os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', '..', 'aws_keys.yaml')
        )
        return s3_saver.S3Saver(
            bucket='dl-checkpoints',
            key='test-cps/cps_test',
            aws_access_key=keys['access'],
            aws_secret_key=keys['secret']
        )

    def _clean(self):
        saver = self._saver()
        for step in saver.global_steps():
            saver.remove_checkpoint(step)

    def test_no_checkpoints(self):
        saver = self._saver()
        self.assertEquals(set(), saver.global_steps())


def _load_yaml(path):
    with open(path, 'r') as f:
        return ruamel.yaml.load(f, Loader=ruamel.yaml.Loader)


if __name__ == '__main__':
    unittest.main()
