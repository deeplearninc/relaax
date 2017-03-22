from relaax.server.common.class_loader import ClassLoader


class TestRLXMessage:

    def test_load_class_from_package(self):
        path = 'tests/fixtures/random_search'
        clazz = ClassLoader.load(path, 'random_search.rs_agent.RSAgent')
        assert clazz().some_method() == "some method result"

    def test_preloaded_module_wouldnt_load_again(self):
        path = 'tests/fixtures/random_search'
        ClassLoader.load(path, 'random_search.rs_agent.RSAgent')
        module = ClassLoader.load_module('random_search', None)
        assert module.__name__ == 'random_search'

    def test_load_class_from_file(self):
        path = 'tests/fixtures/client/rs_client.py'
        clazz = ClassLoader.load(path, 'RSClient')
        assert clazz().some_method() == "some method result"

    def test_load_from_nonexisttent_file_or_wrong_class(self):
        # make sure right type of exceptions raised
        path = 'tests/fixtures/client/sample_other_client.py'
        try:
            ClassLoader.load(path, 'RSClient')
        except ImportError:
            assert True
        path = 'tests/fixtures/client/rs_client.py'
        try:
            ClassLoader.load(path, 'WrongClass')
        except AttributeError:
            assert True
        try:
            path = 'tests/fixtures/random_search'
            ClassLoader.load(path, 'random_search.rs_agent.WrongClass')
        except AttributeError:
            assert True
