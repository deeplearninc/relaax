#!/usr/bin/env python

import numpy
import unittest

from relaax.server.common.bridge.bridge_protocol import BridgeProtocol


class TestBridgeProtocol(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_zero(self):
        messages = list(self.write(0))
        self.assertEquals(len(messages), 1)
        self.assertEquals(self.read(iter(messages)), 0)

    def test_scalars(self):
        for v in (None, False, True, 0, -1, 140, 3.14, 'the string'):
            self.check_protocol(v)

    def test_list(self):
        self.check_protocol([])
        self.check_protocol([None, False, True, 0, -1, 140, 3.14, 'the string'])
        self.check_protocol([[[1, 2], [2, 3]]])

    def test_dict(self):
        self.check_protocol({})
        self.check_protocol({
            'one': None,
            'two': False,
            'three': True,
            'four': 0,
            'five': -1,
            'six': 140,
            'seven': 3.14,
            'eight': 'the string'
        })
        self.check_protocol({
            'one': {
                'two': {
                    'three': 1,
                    'four': 2
                },
                'five': {
                    'six': 2,
                    'seven': 3
                }
            }
        })

    def test_numpy_array(self):
        messages = list(self.write(numpy.array([[1, 2], [3, 4]])))
        self.assertEquals(len(messages), 1)
        self.assertTrue((self.read(iter(messages)) == numpy.array([[1, 2], [3, 4]])).all())

        self.check_protocol(numpy.array([[1, 2], [3, 4.5]]))

    def test_large_numpy_array(self):
        messages = list(self.write(numpy.zeros((1000, 1000), dtype=numpy.float)))
        self.assertEquals(len(messages), 8)
        self.assertTrue((self.read(iter(messages)) == numpy.zeros((1000, 1000), dtype=numpy.float)).all())

    def test_everything(self):
        self.check_protocol({
            'one': [
                'a',
                { 
                    'two': numpy.zeros((1000, 1000)),
                    'three': numpy.array([[1, 2], [3, 4.5]])
                }
            ]
        })

    def write(self, value):
        return BridgeProtocol.build_item_messages(value)

    def read(self, messages):
        return BridgeProtocol.parse_item_messages(messages)

    def check_protocol(self, value):
        self.check_is_equal(value, self.read(self.write(value)))

    def check_is_equal(self, a, b):
        self.assertEquals(type(a), type(b))

        if isinstance(a, list):
            self.assertEquals(len(a), len(b))
            for aa, bb in zip(a, b):
                self.check_is_equal(aa, bb)

        elif isinstance(a, dict):
            self.assertEquals(set(a), set(b))
            for key in a:
                self.assertTrue(isinstance(key, str))
                self.check_is_equal(a[key], b[key])

        elif a is None:
            self.assertEquals(a, b)

        elif isinstance(a, bool):
            self.assertEquals(a, b)

        elif isinstance(a, int):
            self.assertEquals(a, b)

        elif isinstance(a, long):
            self.assertEquals(a, b)

        elif isinstance(a, float):
            self.assertEquals(a, b)

        elif isinstance(a, str):
            self.assertEquals(a, b)

        elif isinstance(a, numpy.ndarray):
            self.assertTrue((a == b).all())

        else:
            assert False


if __name__ == '__main__':
    unittest.main()
