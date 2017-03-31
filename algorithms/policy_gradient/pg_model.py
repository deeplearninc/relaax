from pg_config import config

from relaax.common.algorithms.subgraph import Subgraph

from lib.weights import Weights
from lib.losses import Loss
from lib.networks import FullyConnected
from lib.utils import assemble_and_show_graphs, Placeholder, Placeholders, Assign
from lib.optimizers import ApplyGradients, Adam
from lib.gradients import Gradients
from lib.initializers import Xavier


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


if __name__ == '__main__':
    assemble_and_show_graphs(SharedParameters, PolicyModel)
