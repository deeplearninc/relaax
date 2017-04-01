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
        self.state = Placeholder((None, config.state_size))
        self.action = Placeholder((None, config.action_size))
        self.discounted_reward = Placeholder((None, 1))
        self.weights = Weights()
        self.shared_weights = Placeholders(variables=self.weights)
        self.policy = FullyConnected(state=self.state, weights=self.weights)
        self.loss = Loss(
            action=self.action,
            policy=self.policy,
            discounted_reward=self.discounted_reward
        )
        self.partial_gradients = Gradients(loss=self.loss, variables=self.weights)
        self.assign_weights = Assign(self.weights, self.shared_weights)
        self.initialize = Initialize()


if __name__ == '__main__':
    assemble_and_show_graphs(SharedParameters, PolicyModel)
