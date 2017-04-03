from pg_config import config

from relaax.common.algorithms.subgraph import Subgraph

from lib.graph import Weights, Loss, FullyConnected, Placeholder, Placeholders
from lib.graph import Assign, ApplyGradients, Adam, Gradients, Xavier, Initialize
from lib.utils import assemble_and_show_graphs


# Weights of the policy are shared across
# all agents and stored on the parameter server
class SharedParameters(Subgraph):
    def build(self):
        # Build TF graph
        self.weights = Weights(initializer=Xavier())
        self.gradients = Placeholders(variables=self.weights)
        self.apply_gradients = ApplyGradients(
            Adam(learning_rate=config.learning_rate),
            self.gradients,
            self.weights
        )
        self.initialize = Initialize()


# Policy run by Agent(s)
class PolicyModel(Subgraph):
    def build(self):
        # Build TF graph
        weights = Weights()
        self.ph_weights = self.ph_weights
        self.assign_weights = self.assign_weights

        self.ph_state = Placeholder((None, config.state_size))
        self.policy = FullyConnected(state=self.ph_state, weights=weights.variables)

        self.ph_action = Placeholder((None, config.action_size))
        self.ph_action_probability = Placeholder(...)
        self.ph_discounted_reward = Placeholder((None, 1))
        self.partial_gradients = PartialGradients(
            ph_action=self.ph_action,
            ph_action_probability=self.ph_action_probability,
            ph_discounted_reward=self.ph_discounted_reward, 
            weights=weights.variables)

        self.initialize = Initialize()


if __name__ == '__main__':
    assemble_and_show_graphs(SharedParameters, PolicyModel)
