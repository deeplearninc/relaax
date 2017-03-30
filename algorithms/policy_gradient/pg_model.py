import logging
import numpy as np

from relaax.common.algorithms.decorators import define_scope, define_input

from pg_config import config
from lib.models import BaseModel
from lib.losses import SimpleLoss
from lib.weights import Weights
from lib.networks import FullyConnected
from lib.utils import assemble_and_show_graphs, Placeholder
from lib.optimizers import Adam
from lib.gradients import Gradients
from lib.initializers import Xavier


log = logging.getLogger("policy_gradient")


# Weights of the policy are shared across
# all agents and stored on the parameter server
class SharedParameters(BaseModel):
    def assemble(self):
        # Build TF graph
        self.weights.assemble
        self.gradients.assemble
        self.apply_gradients.assemble

    @define_input
    def weights(self):
        return Weights.assemble(shapes=config.hidden_layers, initializer=Xavier())

    @define_input
    def gradients(self):
        # placeholders to apply gradients to shared parameters
        return Placeholders.assemble(variables=self.weights.ops)

    @define_input
    def apply_gradients(self):
        # apply gradients to weights
        optimizer = Adam(learning_rate=self.config.learning_rate)
        return optimizer.apply_gradients.assemble(self.gradients.ops, self.weights.ops)


# Policy run by Agent(s)
class PolicyModel(BaseModel):
    def assemble(self):
        # Build TF graph
        self.weights.assemble
        self.state.assemble
        self.action.assemble
        self.discounted_reward.assemble
        self.shared_weights.assemble
        self.policy.assemble
        self.partial_gradients.assemble
        self.assign_weights.assemble

    @define_input
    def state(self):
        return Placeholder(np.float32, (None, config.state_size))

    @define_input
    def discounted_reward(self):
        return Placeholder(np.float32)

    @define_scope
    def weights(self):
        return Weights.assemble(shapes=config.hidden_layers)

    @define_scope
    def policy(self):
        return FullyConnected.assemble_from_weights(input=self.state, self.weights.ops)

    @define_scope
    def loss(self):
        return SimpleLoss.assemble(
                output=self.action, weights=self.weights.ops, discounted_reward=self.discounted_reward.op)

    @define_scope
    def partial_gradients(self):
        return Gradients(loss=self.loss, variables=self.weights)

    @define_input
    def shared_weights(self):
        # placeholders to apply weights to shared parameters
        return Placeholders.assemble(variables=self.weights.ops)

    @define_scope
    def assign_weights(self):
        return Assign.assemble(self.weights.ops, self.shared_weights.ops)])


if __name__ == '__main__':
    assemble_and_show_graphs(
        parameter_server=SharedParameters, agent=PolicyModel)
