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
        count, max_episodes = 0, 100
        while count < max_episodes:
            try:
                # if it is terminal state, set terminal to True
                # for example:
                # agent.update(reward=1.1,state=[2.2,3.3],terminal=True)
                # algorithm/state_size and algorithm/action_size to set
                # shape of state and returned action
                action = agent.update(reward=1.1, state=[2.2, 3.3])
                print "action:", action
                print "episode: ", count
                count += 1
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
