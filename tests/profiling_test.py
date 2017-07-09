from __future__ import absolute_import
from builtins import str
from builtins import object

import time
import unittest

from relaax.common import profiling


profiler = profiling.get_profiler(__name__)


class TestProfiling(unittest.TestCase):
    def setUp(self):
        self.handlers = profiling.get_handlers()
        self.mh = MemoryHandler()
        profiling.set_handlers([self.mh])
        print(id(self))

    def tearDown(self):
        profiling.set_handlers(self.handlers)

    def test_global_wrap(self):
        assert len(self.mh.records) == 0
        self.method()
        assert len(self.mh.records) == 0
        profiling.enable(True)
        self.method()
        assert len(self.mh.records) == 1
        self.method()
        assert len(self.mh.records) == 2
        profiling.enable(False)
        self.method()
        assert len(self.mh.records) == 2
        assert self.mh.records[0]['dur'] >= 10000

    def test_wrap(self):
        assert len(self.mh.records) == 0
        self.method2()
        assert len(self.mh.records) == 0
        profiling.enable(True)
        self.method2()
        assert len(self.mh.records) == 1
        self.method2()
        assert len(self.mh.records) == 2
        profiling.enable(False)
        self.method2()
        assert len(self.mh.records) == 2
        assert self.mh.records[0]['cat'] == 'tests.profiling_test'
        assert self.mh.records[0]['name'] == 'method2'
        assert self.mh.records[0]['dur'] >= 10000

    @profiling.wrap
    def method(self):
        time.sleep(0.01)

    @profiler.wrap
    def method2(self):
        time.sleep(0.01)


class MemoryHandler(profiling.Handler):
    def __init__(self):
        super(MemoryHandler, self).__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)
