import numpy as np

from relaax.common.algorithms import subgraph
from relaax.common.algorithms.lib import activation
from relaax.common.algorithms.lib import graph
from relaax.common.algorithms.lib import layer
from relaax.common.algorithms.lib import utils
from lib import da3c_graph
import da3c_config


class Network(subgraph.Subgraph):
    def build_graph(self):
        state = graph.Placeholder(np.float32, shape=[None] +
                da3c_config.options.get('algorithm/state_shape') +
                [da3c_config.options.get('algorithm/history_len')])

        conv1 = layer.Convolution2D(state, 16, 8, 8, subsample=(4, 4),
                border_mode='valid', activation=activation.Relu)
        conv2 = layer.Convolution2D(conv1, 32, 4, 4, subsample=(2, 2),
                border_mode='valid', activation=activation.Relu)

        fc = layer.Dense(graph.Flatten(conv2), 256, activation=activation.Relu)

        actor = layer.Dense(fc, action_size, activation='softmax')
        critic = layer.Dense(fc, 1)

        self.state = state
        self.actor = graph.Softmax(actor)
        self.critic = graph.Flatten(critic)
        self.weights = [layer.weight
                for layer in [conv1, conv2, fc, actor, critic]]


# Weights of the policy are shared across
# all agents and stored on the parameter server
class SharedParameters(subgraph.Subgraph):
    def build_graph(self):
        # Build graph
        ph_increment = graph.Placeholder(np.int64)
        sg_global_step = graph.GlobalStep(ph_increment)
        sg_weights = da3c_graph.Network().weights
        ph_gradients = graph.PlaceholdersByVariables(sg_weights)
        sg_learning_rate = da3c_graph.LearningRate(sg_global_step)
        sg_optimizer = graph.RMSPropOptimizer(
            learning_rate=sg_learning_rate,
            decay=da3c_config.config.RMSProp.decay,
            momentum=0.0,
            epsilon=da3c_config.config.RMSProp.epsilon
        )
        sg_apply_gradients = graph.ApplyGradients(sg_optimizer, sg_weights, ph_gradients)
        sg_initialize = graph.Initialize()

        # Expose public API
        self.op_n_step = self.Op(sg_global_step.n)
        self.op_get_weights = self.Op(sg_weights)
        self.op_apply_gradients = self.Op(sg_apply_gradients, sg_global_step.increment,
            gradients=ph_gradients,
            increment=ph_increment
        )
        self.op_initialize = self.Op(sg_initialize)


# Policy run by Agent(s)
class AgentModel(subgraph.Subgraph):
    def build_graph(self):
        # Build graph
        sg_network = da3c_graph.Network()
        ph_state = sg_network.state
        sg_weights = sg_network.weights
        ph_weights = graph.PlaceholdersByVariables(sg_weights)
        sg_assign_weights = graph.Assign(sg_weights, ph_weights)
        ph_action = graph.Placeholder(np.int32, shape=(None, ))
        ph_value = graph.Placeholder(np.float32, shape=(None, ))
        ph_discounted_reward = graph.Placeholder(np.float32, shape=(None, ))
        sg_loss = da3c_graph.Loss(ph_state, ph_action, ph_value,
                ph_discounted_reward, sg_network.actor, sg_network.critic)
        sg_gradients = graph.Gradients(sg_loss, sg_weights)

        # Expose public API
        self.op_assign_weights = self.Op(sg_assign_weights, weights=ph_weights)
        self.op_get_action_and_value = self.Op(sg_network.actor, sg_network.critic, state=ph_state)
        self.op_compute_gradients = self.Op(sg_gradients,
            state=ph_state,
            action=ph_action,
            value=ph_value,
            discounted_reward=ph_discounted_reward
        )


if __name__ == '__main__':
    utils.assemble_and_show_graphs(SharedParameters, AgentModel)
