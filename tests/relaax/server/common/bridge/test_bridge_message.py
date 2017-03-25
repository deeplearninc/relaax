#!/usr/bin/env python

import numpy
import types
import unittest

from relaax.server.common.bridge.bridge_message import BridgeMessage
from relaax.server.common.bridge.bridge_message_2 import BridgeMessage2


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
        for v in (None, False, True, 0, -1, 140, 3.14, 'the string', numpy.int32(-16)):
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
        messages = list(self.write(numpy.arange(0, 1000000, dtype=numpy.float)))
        self.assertEquals(len(messages), 8)
        self.assertTrue((self.read(iter(messages)) == numpy.arange(0, 1000000, dtype=numpy.float)).all())

    def test_everything(self):
        self.check_protocol({
            'one': [
                'a',
                {
                    'two': numpy.arange(0, 1000000, dtype=numpy.float),
                    'three': numpy.array([[1, 2], [3, 4.5]])
                }
            ]
        })

    def write(self, value):
        return BridgeMessage.serialize(value)

    def read(self, messages):
        return BridgeMessage.deserialize(messages)

    def check_protocol(self, value):
        self.check_are_equal(value, self.read(self.write(value)))

    def check_are_equal(self, a, b):
        self.assertEquals(type(a), type(b))
        {
            list:           self.check_are_lists_equal,
            dict:           self.check_are_dicts_equal,
            types.NoneType: self.assertEquals,
            bool:           self.assertEquals,
            int:            self.assertEquals,
            numpy.int32:    self.assertEquals,
            float:          self.assertEquals,
            str:            self.assertEquals,
            numpy.ndarray:  self.check_are_ndarrays_equal
        }[type(a)](a, b)

    def check_are_lists_equal(self, a, b):
        self.assertEquals(len(a), len(b))
        for aa, bb in zip(a, b):
            self.check_are_equal(aa, bb)

    def check_are_dicts_equal(self, a, b):
        self.assertEquals(set(a), set(b))
        for key in a:
            self.assertTrue(isinstance(key, str))
            self.check_are_equal(a[key], b[key])

    def check_are_ndarrays_equal(self, a, b):
        self.assertTrue((a == b).all())


class TestBridgeProtocol2(TestBridgeProtocol):
    def write(self, value):
        return BridgeMessage2.serialize(value)

    def read(self, messages):
        return BridgeMessage2.deserialize(messages)


if __name__ == '__main__':
    unittest.main()
