import sys


class MockCmdl:

    def __init__(self):
        self.argv = sys.argv
        sys.argv = ['mock_cmdl.py', '--config', 'tests/fixtures/fixture.yaml']

    def restore(self):
        sys.argv = self.argv


cmdl = MockCmdl()
