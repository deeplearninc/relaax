from __future__ import print_function
from builtins import object

import logging
import traceback

from relaax.environment.config import options
from relaax.environment.agent_proxy import AgentProxy, AgentProxyException

log = logging.getLogger(__name__)


class TrainingBase(object):

    def __init__(self):
        self.exploit = options.exploit
        self.max_episodes = options.get('environment/max_episodes', 1)
        self.infinite_run = options.get('environment/infinite_run', False)
        self.agent = AgentProxy(options.rlx_server_address)

    def initialize_agent(self, retry=6):
        # connect to the server
        self.agent.connect(retry)
        # give agent a moment to load and initialize
        self.agent.init(self.exploit)

    def run(self):
        try:
            self.initialize_agent()

            number = 0
            while (number < self.max_episodes) or self.infinite_run:
                try:
                    episode_reward = self.episode(number)
                    if episode_reward is not None:
                        self.agent.metrics.scalar('episode_reward', episode_reward)
                    number += 1

                except AgentProxyException as e:
                    log.error('Agent connection lost: %s' % str(e))
                    log.error('Reconnecting to another Agent, retrying to connect 10 times...')
                    try:
                        self.initialize_agent(retry=10)
                        continue
                    except Exception:
                        raise Exception('Can\'t reconnect, exiting...')

        except Exception as e:
            log.error("Error while running agent: %s" % str(e))
            log.debug(traceback.format_exc())

        finally:
            # disconnect from the server
            self.agent.disconnect()

    def episode(self, number):
        pass
