from pg_config import config

from relaax.common.algorithms.subgraph import Subgraph

from lib.graph import Weights, SimpleLoss, FullyConnected, Placeholder, Placeholders
from lib.graph import Assign, ApplyGradients, Adam, PartialGradients, Xavier, Initialize
from lib.utils import assemble_and_show_graphs


# Weights of the policy are shared across
# all agents and stored on the parameter server
class SharedParameters(Subgraph):
    def build(self):
        # Build TF graph
        self.weights = Weights(initializer=Xavier())
        self.ph_gradients = Placeholders()
        self.apply_gradients = ApplyGradients(
            Adam(learning_rate=config.learning_rate),
            self.ph_gradients,
            self.weights
        )
        self.initialize = Initialize()


# Policy run by Agent(s)
class PolicyModel(Subgraph):
    def build_graph(self):
        # Build graph
        sg_weights = Variables([config.state_size] + config.hidden_sizes + [config.action_size],
                               initializer=Xavier())    # if provided
        ph_weights = Placeholders(sg_weights)

        ph_state = Placeholder((None, config.state_size))
        ph_action = Placeholder((None, config.action_size))
        ph_discounted_reward = Placeholder((None, 1))

        sg_network = FullyConnected(ph_state, sg_weights)
        sg_policy_loss = PolicyLoss(
            action=ph_action,
            discounted_reward=ph_discounted_reward,
            network=sg_network
        )

        sg_policy = Policy(sg_network, sg_policy_loss)

        # Expose public API
        self.op_assign_weights = sg_weights.assign(ph_weights)
        self.op_get_action = sg_policy.get_action(ph_state)
        self.op_compute_gradients = sg_policy.compute_gradients(ph_state, ph_action, ph_discounted_reward)
        self.op_initialize = Initialize()


# Policy run by Agent(s)
class PolicyModelOriginal(Subgraph):
    def build(self):
        # Build TF graph
        ph_weights = Placeholders()
        weights = Weights()
        self.assign_weights = Assign(weights, ph_weights)

        self.ph_state = Placeholder((None, config.state_size))
        self.policy = FullyConnected(self.ph_state, weights)

        self.ph_action = Placeholder((None, config.action_size))
        self.ph_action_probability = Placeholder((None, config.action_size))
        self.ph_discounted_reward = Placeholder((None, 1))
        self.partial_gradients = PartialGradients(
            SimpleLoss(
                action=self.ph_action,
                discounted_reward=self.ph_discounted_reward,
                policy=self.policy # self.ph_state
            ),
            weights
        )

        self.initialize = Initialize()


class Weights(Subgrapg):

    def build_graph(state_size,hidden_sizes,action_size):
        
        self.assign = [
            tf.assign(variable, value)
            for variable, value in zip(variables.node, values.node)
        ]

        shapes = zip([state_size] + hidden_sizes, hidden_sizes + [action_size])
        return [
            tf.Variable(initial_value=initializer(shape=shape, dtype=np.float32))
            for shape in shapes
        ]

    def assign_weights(ph_weights):
        self.build_op(self.assign,weights=ph_weights)


if __name__ == '__main__':
    assemble_and_show_graphs(SharedParameters, PolicyModel)
