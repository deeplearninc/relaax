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


# 'with' decorated
class PolicyModelA(Subgraph):
    def build(self):
        # Build TF graph
        ph_weights = Placeholders()
        weights = Weights()

        with Params(ph_weights):
            self.assign_weights = Assign(weights, ph_weights)

        state = Placeholder((None, config.state_size))

        with Params(state):
            self.policy = FullyConnected(state, weights)

        action = Placeholder((None, config.action_size))
        action_probability = Placeholder((None, config.action_size))
        self.discounted_reward = Placeholder((None, 1))

        with Params(state, action, action_probability, discounted_reward):
            self.partial_gradients = PartialGradients(
                SimpleLoss(
                    action=action,
                    discounted_reward=discounted_reward,
                    policy=self.policy # state
                ),
                weights
            )

        self.initialize = Initialize()


# 'entry' calls
class PolicyModelB(Subgraph):
    def build(self):
        # Build TF graph
        ph_weights = Placeholders()
        weights = Weights()

        self.assign_weights = Entry(ph_weights, Assign(weights, ph_weights))

        state = Placeholder((None, config.state_size))

        self.policy = Entry(state, FullyConnected(state, weights))

        action = Placeholder((None, config.action_size))
        action_probability = Placeholder((None, config.action_size))
        discounted_reward = Placeholder((None, 1))

        self.partial_gradients = Entry(state, action, action_probability, discounted_reward,
            PartialGradients(
                SimpleLoss(
                    action=action,
                    discounted_reward=discounted_reward,
                    policy=self.policy # state
                ),
                weights
            )
        )

        self.initialize = Initialize()


# 'entry' calls with named parameters
class PolicyModelC(Subgraph):
    def build(self):
        # Build TF graph
        ph_weights = Placeholders()
        weights = Weights()

        self.assign_weights = Entry(Assign(weights, ph_weights), weights=ph_weights)

        state = Placeholder((None, config.state_size))

        self.policy = Entry(FullyConnected(state, weights), state=state)

        action = Placeholder((None, config.action_size))
        action_probability = Placeholder((None, config.action_size))
        discounted_reward = Placeholder((None, 1))

        self.partial_gradients = Entry(
            PartialGradients(
                SimpleLoss(
                    action=action,
                    discounted_reward=discounted_reward,
                    policy=self.policy # state
                ),
                weights
            ),
            state=state,
            action=action,
            action_probability=action_probability,
            discounted_reward=discounted_reward
        )

        self.initialize = Initialize()


# 'def' calls
class PolicyModelD(Subgraph):
    def build(self):
        # Build TF graph
        ph_weights = Placeholders()
        weights = Weights()

        Def('assign_weights', ph_weights, Assign(weights, ph_weights))

        state = Placeholder((None, config.state_size))

        Def('policy', state, FullyConnected(state, weights))

        action = Placeholder((None, config.action_size))
        action_probability = Placeholder((None, config.action_size))
        discounted_reward = Placeholder((None, 1))

        Def('partial_gradients', state, action, action_probability, discounted_reward,
            PartialGradients(
                SimpleLoss(
                    action=action,
                    discounted_reward=discounted_reward,
                    policy=self.policy # state
                ),
                weights
            )
        )

        Def('initialize', Initialize())


# 'def' calls on explicit defs
class PolicyModelE(Subgraph):
    def build(self):
        # Build TF graph
        ph_weights = Placeholders()
        weights = Weights()

        Def(self.assign_weights, ph_weights, Assign(weights, ph_weights))

        state = Placeholder((None, config.state_size))

        Def(self.policy, state, FullyConnected(state, weights))

        action = Placeholder((None, config.action_size))
        action_probability = Placeholder((None, config.action_size))
        discounted_reward = Placeholder((None, 1))

        Def(self.partial_gradients, state, action, action_probability, discounted_reward,
            PartialGradients(
                SimpleLoss(
                    action=action,
                    discounted_reward=discounted_reward,
                    policy=self.policy # state
                ),
                weights
            )
        )

        self.initialize = Initialize()

    def assign_weights(ph_weights):
        pass

    def policy(state):
        pass

    def partial_gradients(state, action, action_probability, discounted_reward):
        pass


# 'build_op' calls on external functions
class PolicyModelF(Subgraph):
    def build_graph(self):
        # Build TF graph

        sg_weights = Variables([config.state_size] + config.hidden_sizes + [config.action_size])
        ph_weights = Placeholders(sg_weights)

        ph_state = Placeholder((None, config.state_size))

        sg_policy = Policy(ph_state, sg_weights)

        ph_action = Placeholder((None, config.action_size))
        ph_discounted_reward = Placeholder((None, 1))

        sg_gradients = Gradients(
            PolicyLoss(
                action=ph_action,
                discounted_reward=ph_discounted_reward,
                policy=sg_policy
            ),
            sg_weights
        )

        self.op_assign_weights = sg_weights.assign(ph_weights)
        self.op_get_action = sg_policy.get_action(ph_state)
        self.op_compute_gradients = sg_gradients.compute(ph_state, ph_action, ph_discounted_reward)
        self.op_initialize = Initialize()


# 'build_op' calls
class PolicyModelG(Subgraph):
    def build_graph(self):
        # Build TF graph
        ph_weights = Placeholders()
        weights = Weights()

        assign = Assign(weights, ph_weights)

        state = Placeholder((None, config.state_size))

        policy = FullyConnected(state, weights)

        action = Placeholder((None, config.action_size))
        action_probability = Placeholder((None, config.action_size))
        discounted_reward = Placeholder((None, 1))

        partial_gradients = PartialGradients(
            SimpleLoss(
                action=action,
                discounted_reward=discounted_reward,
                policy=policy # state
            ),
            weights
        )

        initialize = Initialize()

        self.assign_weights = self.build_op(assign, ph_weights)
        self.policy = self.build_op(policy, state=state)
        self.partial_gradients = self.build_op(partial_gradients, state, action, action_probability, discounted_reward)
        self.initialize = self.build_op(initialize)



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
