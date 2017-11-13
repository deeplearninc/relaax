from __future__ import absolute_import

import logging
import numpy as np

from relaax.common.algorithms import subgraph
from relaax.common.algorithms.lib import graph
from relaax.common.algorithms.lib import layer
from relaax.common.algorithms.lib import optimizer
from relaax.common.algorithms.lib import loss
from relaax.common.algorithms.lib import utils

from relaax.algorithms.trpo.trpo_model import GetVariablesFlatten, SetVariablesFlatten

from . import pg_config

logger = logging.getLogger(__name__)


class Network(subgraph.Subgraph):
    def build_graph(self):
        input = layer.ConfiguredInput(pg_config.config.input)

        dense = layer.GenericLayers(layer.Flatten(input),
                                    [dict(type=layer.Dense, size=size, activation=layer.Activation.Relu)
                                    for size in pg_config.config.hidden_sizes])

        actor = layer.Dense(dense, pg_config.config.output.action_size,
                            activation=layer.Activation.Softmax)

        self.state = input.ph_state
        self.weights = layer.Weights(input, dense, actor)
        return actor.node


# Weights of the policy are shared across
# all agents and stored on the parameter server
class SharedParameters(subgraph.Subgraph):
    def build_graph(self):
        # Build graph
        sg_global_step = graph.GlobalStep()
        sg_weights = Network().weights
        sg_optimizer = optimizer.AdamOptimizer(pg_config.config.learning_rate)
        sg_gradients = optimizer.Gradients(sg_weights, optimizer=sg_optimizer)
        sg_initialize = graph.Initialize()

        # Weights get/set for updating the policy
        sg_get_weights_flatten = GetVariablesFlatten(sg_weights)
        sg_set_weights_flatten = SetVariablesFlatten(sg_weights)

        # Expose public API
        self.op_n_step = self.Op(sg_global_step.n)
        self.op_get_weights = self.Op(sg_weights)
        self.op_get_weights_signed = self.Ops(sg_weights, sg_global_step.n)

        self.op_apply_gradients = self.Ops(sg_gradients.apply,
                                           sg_global_step.increment, gradients=sg_gradients.ph_gradients,
                                           increment=sg_global_step.ph_increment)

        self.op_get_weights_flatten = self.Op(sg_get_weights_flatten)
        self.op_set_weights_flatten = self.Op(sg_set_weights_flatten, value=sg_set_weights_flatten.ph_value)

        # Gradient combining routines

        # First come, first served gradient update
        def func_fifo_gradient(session, gradients, step):
            current_step = session.op_n_step()
            logger.info("Gradient with step {} received from agent. Current step: {}".format(step, current_step))
            session.op_apply_gradients(gradients=gradients, increment=1)

        # Accumulate gradients from many agents and average them
        self.gradients = []

        def func_average_gradient(session, gradients, step):
            logger.info("received a gradient, number of gradients collected so far: {}".format(len(self.gradients)))
            if step >= session.op_n_step():
                logger.info("gradient is fresh, accepted")
                self.gradients.append(gradients)
            else:
                logger.info("gradient is old, rejected")

            if len(self.gradients) >= pg_config.config.num_gradients:
                # we have collected enough gradients, no we can average them and make a step
                logger.info("computing mean grad")
                flat_grads = [Shaper.get_flat(g) for g in self.gradients]
                mean_flat_grad = np.mean(np.stack(flat_grads), axis=0)

                mean_grad = Shaper.reverse(mean_flat_grad, gradients)
                session.op_apply_gradients(gradients=mean_grad, increment=1)
                self.gradients = []

        # Asynchronous Stochastic Gradient Descent with Delay Compensation, see https://arxiv.org/pdf/1609.08326.pdf
        self.weights_history = {}

        def init_weight_history(session):
            self.weights_history[0] = session.op_get_weights_flatten()

        self.op_init_weight_history = self.Call(init_weight_history)

        def func_dc_gradient(session, gradients, step):
            # Assume step to be global step number
            current_step = session.op_n_step()
            current_weights_f = session.op_get_weights_flatten()

            old_weights_f = self.weights_history.get(step, current_weights_f)

            new_gradient_f = Shaper.get_flat(gradients)

            # Compute new gradient
            delta = pg_config.config.dc_lambda * (
                new_gradient_f * new_gradient_f * (current_weights_f - old_weights_f))
            compensated_gradient_f = new_gradient_f + delta

            compensated_gradient = Shaper.reverse(compensated_gradient_f, gradients)

            session.op_apply_gradients(gradients=compensated_gradient, increment=1)
            updated_weights = session.op_get_weights_flatten()
            updated_step = session.op_n_step()
            self.weights_history[updated_step] = updated_weights

            # Cleanup history
            for k in list(self.weights_history.keys()):
                if k < updated_step - 20:
                    try:
                        del self.weights_history[k]
                    except KeyError:
                        pass

        if pg_config.config.combine_gradient == 'fifo':
            self.op_submit_gradients = self.Call(func_fifo_gradient)
        elif pg_config.config.combine_gradient == 'average':
            self.op_submit_gradients = self.Call(func_average_gradient)
        elif pg_config.config.combine_gradient == 'dc':
            self.op_submit_gradients = self.Call(func_dc_gradient)
        else:
            logger.error("Unknown gradient combination mode: {}".format(pg_config.config.combine_gradient))

        self.op_initialize = self.Op(sg_initialize)


# Policy run by Agent(s)
class PolicyModel(subgraph.Subgraph):
    def build_graph(self):
        # Build graph
        sg_network = Network()

        sg_loss = loss.PGLoss(action_size=pg_config.config.output.action_size,
                              network=sg_network)
        sg_gradients = optimizer.Gradients(sg_network.weights, loss=sg_loss)

        # Expose public API
        self.op_assign_weights = self.Op(sg_network.weights.assign,
                                         weights=sg_network.weights.ph_weights)
        self.op_get_action = self.Op(sg_network, state=sg_network.state)
        self.op_compute_gradients = self.Op(sg_gradients.calculate,
                                            state=sg_network.state, action=sg_loss.ph_action,
                                            discounted_reward=sg_loss.ph_discounted_reward)


class Shaper():
    @staticmethod
    def numel(x):
        return np.prod(np.shape(x))

    @classmethod
    def get_flat(cls, v):
        tensor_list = list(utils.Utils.flatten(v))
        u = np.concatenate([np.reshape(t, newshape=[cls.numel(t), ]) for t in tensor_list], axis=0)
        return u

    @staticmethod
    def reverse(u, base_shape):
        tensor_list = list(utils.Utils.flatten(base_shape))
        shapes = map(np.shape, tensor_list)
        v_flat = []
        start = 0
        for (shape, t) in zip(shapes, tensor_list):
            size = np.prod(shape)
            v_flat.append(np.reshape(u[start:start + size], shape))
            start += size
        v = utils.Utils.reconstruct(v_flat, base_shape)
        return v

if __name__ == '__main__':
    utils.assemble_and_show_graphs(SharedParameters, PolicyModel)
