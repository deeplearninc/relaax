import pg_config
from lib import pg_episode


# PGAgent implements training regime for Policy Gradient algorithm
# If exploit on init set to True, agent will run in exploitation regime:
# stop updating shared parameters and at the end of every episode load
# new policy parameters from PS
class PGAgent(object):
    def __init__(self, parameter_server):
        self.ps = parameter_server

    # environment is ready and
    # waiting for agent to initialize
    def init(self, exploit=False):
        self.episode = pg_episode.PGEpisode(self.ps, exploit)
        self.episode.begin()
        return True

    # environment generated new state and reward
    # and asking agent for an action for this state
    def update(self, reward, state, terminal):
        action = self.episode.step(reward, state, terminal)

        if (self.episode.experience_size == pg_config.config.batch_size) or terminal:
            self.episode.end()
            self.episode.begin()

        return action


    # environment is asking to reset agent
    def reset(self):
        self.episode.reset()
        return True
