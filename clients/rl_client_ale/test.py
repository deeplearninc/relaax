import unittest

import main

class TestServerAPI(unittest.TestCase):
    def setUp(self):
        s = main.ServerAPI()

    def test_on_join_ack(self):
        self.assertTrue(True)

if __name__ == "__main__":
    unittest.main()
    
