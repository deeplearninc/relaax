from builtins import object
from relaax.server.common.class_loader import ClassLoader


class TestRLXMessage(object):

    def test_load_class_from_package(self):
        path = 'tests/fixtures/an_app/random_search'
        clazz = ClassLoader.load(path, 'random_search.rs_agent.RSAgent')
        assert clazz().some_method() == "some method result"

    def test_preloaded_module_wouldnt_load_again(self):
        path = 'tests/fixtures/an_app/random_search'
        ClassLoader.load(path, 'random_search.rs_agent.RSAgent')
        module = ClassLoader.load_module('random_search', None)
        assert module.__name__ == 'random_search'

    def test_load_class_from_file(self):
        path = 'tests/fixtures/an_app/client/rs_client.py'
        clazz = ClassLoader.load(path, 'RSClient')
        assert clazz().some_method() == "some method result"

    def test_load_from_nonexisttent_file_or_wrong_class(self):
        # make sure right type of exceptions raised
        path = 'tests/fixtures/an_app/client/sample_other_client.py'
        try:
            ClassLoader.load(path, 'RSClient')
            assert False
        except ImportError:
            assert True
        path = 'tests/fixtures/an_app/client/rs_client.py'
        try:
            ClassLoader.load(path, 'WrongClass')
            assert False
        except AttributeError:
            assert True
        try:
            path = 'tests/fixtures/an_app/random_search'
            ClassLoader.load(path, 'random_search.rs_agent.WrongClass')
            assert False
        except AttributeError:
            assert True
