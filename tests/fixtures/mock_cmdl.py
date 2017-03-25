import sys


class MockCmdl:

    def __init__(self):
        self.argv = sys.argv
        self.set_args(['mock_cmdl.py', '--config', 'tests/fixtures/fixture.yaml'])

    def set_args(self, argv):
        sys.argv = argv

    def restore(self):
        sys.argv = self.argv


cmdl = MockCmdl()
