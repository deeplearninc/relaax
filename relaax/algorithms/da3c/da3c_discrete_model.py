from __future__ import absolute_import

import logging
import tensorflow as tf

from relaax.common.algorithms import subgraph
from relaax.common.algorithms.lib import graph
from relaax.common.algorithms.lib import layer
from relaax.common.algorithms.lib import loss
from relaax.common.algorithms.lib import optimizer
from relaax.common.algorithms.lib import lr_schedule
from . import da3c_config
from . import icm_model

logger = logging.getLogger(__name__)


class Network(subgraph.Subgraph):
    def build_graph(self):
        input = layer.ConfiguredInput(da3c_config.config.input)

        sizes = da3c_config.config.hidden_sizes
        layers = [input]
        flattened_input = layer.Flatten(input)

        last_size = flattened_input.node.shape.as_list()[-1]
        if len(sizes) > 0:
            last_size = sizes[-1]

        if da3c_config.config.use_lstm:
            lstm = layer.LSTM(graph.Expand(flattened_input, 0), n_units=last_size)
            head = graph.Reshape(lstm, [-1, last_size])
            layers.append(lstm)

            self.ph_lstm_state = lstm.ph_state
            self.lstm_zero_state = lstm.zero_state
            self.lstm_state = lstm.state
        else:
            activation = layer.Activation.get_activation(da3c_config.config.activation)
            head = layer.GenericLayers(flattened_input,
                                       [dict(type=layer.Dense, size=size,
                                             activation=activation) for size in sizes])
            layers.append(head)

        actor = layer.Actor(head, da3c_config.config.output)
        critic = layer.Dense(head, 1)
        layers.extend((actor, critic))

        self.ph_state = input.ph_state
        self.actor = actor
        self.critic = graph.Flatten(critic)
        self.weights = layer.Weights(*layers)
        self.actor_weights = layer.Weights(actor)


# Weights of the policy are shared across
# all agents and stored on the parameter server
class SharedParameters(subgraph.Subgraph):
    def build_graph(self):
        # Build graph
        sg_global_step = graph.GlobalStep()
        sg_network = Network()
        sg_weights = sg_network.weights

        if da3c_config.config.optimizer == 'Adam':
            sg_optimizer = optimizer.AdamOptimizer(da3c_config.config.initial_learning_rate)
        else:
            sg_learning_rate = lr_schedule.Linear(sg_global_step,
                                                  da3c_config.config.initial_learning_rate,
                                                  da3c_config.config.max_global_step)
            sg_optimizer = optimizer.RMSPropOptimizer(learning_rate=sg_learning_rate,
                                                      decay=da3c_config.config.RMSProp.decay, momentum=0.0,
                                                      epsilon=da3c_config.config.RMSProp.epsilon)
        sg_gradients = optimizer.Gradients(sg_weights, optimizer=sg_optimizer)

        if da3c_config.config.use_icm:
            sg_icm_optimizer = optimizer.AdamOptimizer(da3c_config.config.icm.lr)
            sg_icm_weights = icm_model.ICM().weights
            sg_icm_gradients = optimizer.Gradients(sg_icm_weights, optimizer=sg_icm_optimizer)

            # Expose ICM public API
            self.op_icm_get_weights = self.Op(sg_icm_weights)
            self.op_icm_apply_gradients = self.Op(sg_icm_gradients.apply,
                                                  gradients=sg_icm_gradients.ph_gradients)

        sg_initialize = graph.Initialize()

        # Expose public API
        self.op_n_step = self.Op(sg_global_step.n)
        self.op_check_weights = self.Ops(sg_weights.check, sg_global_step.n)
        self.op_get_weights = self.Op(sg_weights)
        self.op_apply_gradients = self.Ops(sg_gradients.apply, sg_global_step.increment,
                                           gradients=sg_gradients.ph_gradients,
                                           increment=sg_global_step.ph_increment)

        # Gradients' applying methods: fifo (by default), averaging, delay compensation

        # First come, first served gradient update
        def func_fifo_gradient(session, gradients, step_inc, agent_step):
            global_step = session.op_n_step()
            logger.debug("Gradient with step {} received from agent. Current step: {}".format(agent_step,
                                                                                              global_step))
            session.op_apply_gradients(gradients=gradients, increment=step_inc)

        # Accumulate gradients from many agents and average them
        self.gradients = []

        def func_average_gradient(session, gradients, step_inc, agent_step):
            pass

        # Asynchronous Stochastic Gradient Descent with Delay Compensation,
        # see https://arxiv.org/pdf/1609.08326.pdf
        self.weights_history = {}

        def init_weight_history(session):
            self.weights_history[0] = session.op_get_weights_flatten()

        self.op_init_weight_history = self.Call(init_weight_history)

        def func_dc_gradient(session, gradients, step_inc, agent_step):
            pass

        if da3c_config.config.combine_gradients == 'fifo':
            self.op_submit_gradients = self.Call(func_fifo_gradient)
        elif da3c_config.config.combine_gradient == 'average':
            self.op_submit_gradients = self.Call(func_average_gradient)
        elif da3c_config.config.combine_gradient == 'dc':
            self.op_submit_gradients = self.Call(func_dc_gradient)
        else:
            logger.error("Unknown gradient combination mode: {}".format(da3c_config.config.combine_gradients))

        self.op_initialize = self.Op(sg_initialize)


# Policy run by Agent(s)
class AgentModel(subgraph.Subgraph):
    def build_graph(self):
        # Build graph
        sg_network = Network()

        sg_loss = loss.DA3CLoss(sg_network.actor, sg_network.critic, da3c_config.config)
        sg_gradients = optimizer.Gradients(sg_network.weights, loss=sg_loss,
                                           norm=da3c_config.config.gradients_norm_clipping)

        if da3c_config.config.use_icm:
            sg_icm_network = icm_model.ICM()
            sg_icm_gradients = optimizer.Gradients(sg_icm_network.weights, loss=sg_icm_network.loss)

            # Expose ICM public API
            self.op_icm_assign_weights = self.Op(sg_icm_network.weights.assign,
                                                 weights=sg_icm_network.weights.ph_weights)

            feeds = dict(state=sg_icm_network.ph_state, probs=sg_icm_network.ph_probs)
            self.op_get_intrinsic_reward = self.Ops(sg_icm_network.rew_out, **feeds)

            feeds.update(dict(action=sg_icm_network.ph_taken))
            self.op_compute_icm_gradients = self.Op(sg_icm_gradients.calculate, **feeds)

        batch_size = tf.to_float(tf.shape(sg_network.ph_state.node)[0])

        summaries = tf.summary.merge([
            tf.summary.scalar('policy_loss', sg_loss.policy_loss / batch_size),
            tf.summary.scalar('value_loss', sg_loss.value_loss / batch_size),
            tf.summary.scalar('entropy', sg_loss.entropy / batch_size),
            tf.summary.scalar('gradients_global_norm', sg_gradients.global_norm),
            tf.summary.scalar('weights_global_norm', sg_network.weights.global_norm)])

        # Expose public API
        self.op_assign_weights = self.Op(sg_network.weights.assign, weights=sg_network.weights.ph_weights)

        feeds = dict(state=sg_network.ph_state, action=sg_loss.ph_action,
                     advantage=sg_loss.ph_advantage, discounted_reward=sg_loss.ph_discounted_reward)

        if da3c_config.config.use_lstm:
            feeds.update(dict(lstm_state=sg_network.ph_lstm_state))
            self.lstm_zero_state = sg_network.lstm_zero_state
            self.op_get_action_value_and_lstm_state = self.Ops(sg_network.actor, sg_network.critic,
                                                               sg_network.lstm_state,
                                                               state=sg_network.ph_state,
                                                               lstm_state=sg_network.ph_lstm_state)
        else:
            self.op_get_action_and_value = self.Ops(sg_network.actor, sg_network.critic,
                                                    state=sg_network.ph_state)

        self.op_compute_gradients_and_summaries = self.Ops(sg_gradients.calculate, summaries, **feeds)
