import random

from relaax.client.rlx_client_config import options
from relaax.client.rlx_client import RlxClient, RlxClientException


def sample_exchange():
    agent = RlxClient(options.get('relaax_rlx_server/bind', 'localhost:7001'))

    try:
        # connect to the server
        agent.connect()

        # give agent a moment to load and initialize
        res = agent.init()
        print "on init: ", res

        # update agent with state and reward
        for count in xrange(100):
            try:
                reward = None
                for step in xrange(10):
                    if random.random() >= 0.5:
                        state = [1, 0]
                    else:
                        state = [0, 1]
                    action = agent.update(reward=reward, state=state, terminal=False)
                    action = action['data']
                    print "action/episode/step:", action, count, step
                    reward = (action - state[0]) ** 2
                agent.update(reward=reward, state=None, terminal=True)
            except RlxClientException as e:
                print "agent connection lost: ", e
                print ('reconnecting to another agent, '
                       'retrying to connect 10 times before giving up...')
                try:
                    agent.connect(retry=10)
                    agent.init()
                    continue
                except:
                    print "Can't reconnect, exiting"
                    raise Exception("Can't reconnect")

        # reset agent
        res = agent.reset()
        print "on reset:", res

    except Exception as e:
        print "Something went wrong: ", e

    finally:
        # disconnect from the server
        if agent:
            agent.disconnect()


if __name__ == '__main__':
    sample_exchange()
