from __future__ import print_function
from builtins import range
from builtins import object
from relaax.client.rlx_client_config import options
from relaax.client.rlx_client import RlxClient, RlxClientException

from bandit import Bandit


class Training(object):

    def __init__(self):
        self.agent = RlxClient(options.get('relaax_rlx_server/bind', 'localhost:7001'))
        self.steps = 5000
        self.bandit = Bandit()

    def run(self):
        try:
            # connect to the server
            self.agent.connect()
            # give agent a moment to load and initialize
            self.agent.init(options.get('exploit', False))
            # get first action from agent
            action = self.agent.update(reward=None, state=[])
            # update agent with state and reward
            for step in range(self.steps):
                try:
                    reward = self.bandit.pull(action)
                    action = self.agent.update(reward=reward, state=[])
                    print('step: %s, action: %s' % (step, action))
                except RlxClientException as e:
                    print("agent connection lost: ", e)
                    print ('reconnecting to another agent, '
                           'retrying to connect 10 times before giving up...')
                    try:
                        self.agent.connect(retry=10)
                        self.agent.init()
                        continue
                    except:
                        print("Can't reconnect, exiting")
                        raise Exception("Can't reconnect")

        except Exception as e:
            print("Something went wrong: ", e)

        finally:
            # disconnect from the server
            self.agent.disconnect()


if __name__ == '__main__':
    Training().run()
