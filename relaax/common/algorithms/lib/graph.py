from __future__ import print_function
from builtins import object

import logging
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import init_ops
from tensorflow.contrib.rnn import RNNCell

from relaax.common.algorithms.lib import utils
from relaax.common.algorithms import subgraph

logger = logging.getLogger(__name__)


class DefaultInitializer(object):
    INIT = {
        np.float32: (tf.float32, init_ops.glorot_uniform_initializer)
    }

    def __call__(self, dtype=np.float32, shape=None):
        tf_dtype, initializer = self.INIT[dtype]
        return initializer(dtype=tf_dtype)(shape=shape, dtype=tf_dtype)


class ZeroInitializer(object):
    def __call__(self, dtype=np.float32, shape=None):
        return np.zeros(shape=shape, dtype=dtype)


class OneInitializer(object):
    def __call__(self, dtype=np.float32, shape=None):
        return np.ones(shape=shape, dtype=dtype)


class RandomUniformInitializer(object):
    DTYPE = {
        np.float: tf.float64,
        np.float64: tf.float64,
        np.float32: tf.float32,
    }

    def __init__(self, minval=0, maxval=1):
        self.minval = minval
        self.maxval = maxval

    def __call__(self, dtype=np.float32, shape=None):
        return tf.random_uniform(
            shape,
            dtype=self.DTYPE[dtype],
            minval=self.minval,
            maxval=self.maxval
        )


class RandomNormalInitializer(object):
    DTYPE = {
        np.float: tf.float64,
        np.float64: tf.float64,
        np.float32: tf.float32,
    }

    def __init__(self, mean=0.0, stddev=1.0):
        self.mean = mean
        self.stddev = stddev

    def __call__(self, dtype=np.float32, shape=None):
        return tf.random_normal(
            shape,
            dtype=self.DTYPE[dtype],
            mean=self.mean,
            stddev=self.stddev,
        )


class XavierInitializer(object):
    DTYPE = {
        np.float: tf.float64,
        np.float64: tf.float64,
        np.float32: tf.float32,
    }

    def __call__(self, dtype=np.float32, shape=None):
        return tf.contrib.layers.xavier_initializer()(
            dtype=self.DTYPE[dtype],
            shape=shape
        )


class L2loss(subgraph.Subgraph):
    """Computes half the L2 norm of a tensor without the sqrt."""

    def build_graph(self, t, name=None):
        """
        Args:
            t: A Tensor.
            name: A name for the operation (optional).

        Returns:
            A Tensor. Has the same type as t.
        """
        self.op = tf.nn.l2_loss(t, name=name)


class SparseSoftmaxCrossEntropyWithLogits(subgraph.Subgraph):
    """Computes sparse softmax cross entropy between `logits` and `labels`."""

    def build_graph(self, logits, labels, name=None):
        """
        Args:
          logits: Unscaled log probabilities of rank `r` and shape
            `[d_0, d_1, ..., d_{r-2}, num_classes]` with `float` dtype.
          labels: `Tensor` of shape `[d_0, d_1, ..., d_{r-2}]` with `int` dtype.
            Each entry in `labels` must be an index in `[0, num_classes)`.
          name: A name for the operation (optional).

        Returns:
          A `Tensor` of the same shape as `labels` and of the same type as `logits`
          with the softmax cross entropy loss.
        """
        self.op = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits.node, labels=labels.node), name=name)


class ArgMax(subgraph.Subgraph):
    def build_graph(self, input, axis=None, name=None):
        self.op = tf.argmax(input.node, axis=axis, name=name)


class Softmax(subgraph.Subgraph):
    def build_graph(self, x):
        return tf.nn.softmax(x.node)


class Reshape(subgraph.Subgraph):
    def build_graph(self, x, shape):
        return tf.reshape(x.node, shape)


class Flatten(subgraph.Subgraph):
    def build_graph(self, x):
        return Reshape(x, (-1,)).node


class Expand(subgraph.Subgraph):
    def build_graph(self, x, axis=0):
        return tf.expand_dims(x.node, axis=axis)


class Concat(subgraph.Subgraph):
    def build_graph(self, values, axis, name='concat'):
        return tf.concat([v.node for v in values], axis, name=name)


class List(subgraph.Subgraph):
    def build_graph(self, items):
        self.items = list(items)
        return [i.node for i in self.items]


class Assign(subgraph.Subgraph):
    def build_graph(self, variables, values):
        return [
            tf.assign(variable, value)
            for variable, value in utils.Utils.izip(
                variables.node,
                values.node
            )
        ]


class Increment(subgraph.Subgraph):
    def build_graph(self, variable, increment):
        return tf.assign_add(variable.node, increment.node)


class VarAssign(subgraph.Subgraph):
    def build_graph(self, variable, value):
        self.ph_variable = Placeholders(variables=TfNode(variable))
        self.assign_from_ph = TfNode(tf.assign(variable, self.ph_variable.checked))
        self.assign_from_value = TfNode(tf.assign(variable, tf.constant(value)))
        return variable


class Constant(subgraph.Subgraph):
    """Creates a constant tensor."""

    def build_graph(self, value, dtype=None, shape=None, name='Const'):
        """
        Args:
            value: A constant value (or list) of output type dtype.
            dtype: The type of the elements of the resulting tensor.
            shape: Optional dimensions of resulting tensor.
            name: Optional name for the tensor.

        Returns:
            A Constant Tensor.
        """

        return tf.constant(value, dtype=dtype, shape=shape, name=name)


class Placeholder(subgraph.Subgraph):
    """Placeholder of given shape."""

    DTYPE = {
        np.int32: tf.int32,
        np.int64: tf.int64,
        np.float32: tf.float32,
        np.float64: tf.float64
    }

    def build_graph(self, dtype, shape=None, name=None):
        """Assemble one placeholder.

        Args:
            shape: The shape of the tensor to be fed (optional). If the shape is not
      specified, you can feed a tensor of any shape.
            dtype: The type of elements in the placeholder to be fed.
            name: A name for the placeholder (optional).

        Returns:
            placeholder of given shape and data type
        """

        ph = tf.placeholder(self.DTYPE[dtype], shape=shape, name=name)
        if dtype not in [np.int32, np.int64]:
            self.checked = tf.check_numerics(ph, '')
        return ph


class Placeholders(subgraph.Subgraph):
    def build_graph(self, variables):
        phs = utils.Utils.map(variables.node, lambda v: tf.placeholder(shape=v.get_shape(), dtype=v.dtype))
        self.checked = utils.Utils.map(phs, lambda ph: tf.check_numerics(ph, ''))
        return phs


class GlobalStep(subgraph.Subgraph):
    def build_graph(self):
        self.n = Variable(0, dtype=np.int64)
        self.ph_increment = Placeholder(np.int64)
        self.increment = Increment(self.n, self.ph_increment)


class Variable(subgraph.Subgraph):
    DTYPE = {
        None: None,
        np.int32: tf.int32,
        np.int64: tf.int64,
        np.float64: tf.float64
    }

    def build_graph(self, initial_value, dtype=None):
        return tf.Variable(initial_value, dtype=self.DTYPE[dtype])


class Variables(subgraph.Subgraph):
    def build_graph(self, *variables):
        return [variable.node for variable in variables]

    def assign(self, values):
        return TfNode([
            tf.assign(variable, value)
            for variable, value in utils.Utils.izip(
                self.node,
                values.node
            )
        ])


class Initialize(subgraph.Subgraph):
    def build_graph(self):
        return tf.global_variables_initializer()


class TfNode(subgraph.Subgraph):
    def build_graph(self, tf_tensor):
        return tf_tensor


class AssignWeights(subgraph.Subgraph):
    def build_graph(self, w1, w2, part=None):
        if part is None:
            self.op = TfNode([tf.assign(variable, value) for variable, value
                              in utils.Utils.izip(w1.node, w2.node)])
        else:
            trap = 1. - part
            self.op = TfNode([tf.assign(variable, trap * variable + part * value) for variable, value
                              in utils.Utils.izip(w1.node, w2.node)])


class LinearMovingAverage(subgraph.Subgraph):
    def build_graph(self, size):
        sum_ = tf.Variable(0, dtype=np.float64)
        count = tf.Variable(0, dtype=np.float64)

        pointer = tf.Variable(0, dtype=np.int32)
        sums = tf.Variable([0] * size, dtype=np.float64)
        counts = tf.Variable([0] * size, dtype=np.float64)

        ph_sum = Placeholder(np.float64)
        ph_count = Placeholder(np.float64)

        update_sum = tf.assign_add(sum_, ph_sum.node - sums[pointer])
        update_count = tf.assign_add(count, ph_count.node - counts[pointer])

        with tf.get_default_graph().control_dependencies([update_sum]):
            update_sums = tf.scatter_update(sums, [pointer], [ph_sum.node])

        with tf.get_default_graph().control_dependencies([update_count]):
            update_counts = tf.scatter_update(counts, [pointer], [ph_count.node])

        with tf.get_default_graph().control_dependencies([update_sum, update_count, update_sums,
                                                          update_counts]):
            move = tf.assign(pointer, (pointer + 1) % size)

        self.ph_sum = ph_sum
        self.ph_count = ph_count
        self.add = TfNode(tf.group(update_sum, update_count, update_sums, update_counts, move))
        self.average = TfNode(sum_ / tf.maximum(tf.cast(1e-10, tf.float64), count))


def get_gradients_apply_routine(cfg):
    if cfg.combine_gradients == 'fifo':
        return GradientFIFO().func
    elif cfg.combine_gradients == 'avg':
        return GradientAVG(cfg).func
    elif cfg.combine_gradients == 'dc':
        return GradientDC(cfg).func
    elif cfg.combine_gradients == 'all':
        return GradientALL(cfg).func
    else:
        logger.error("Unknown gradient combination mode: {}".format(cfg.combine_gradients))


class GradientFIFO(subgraph.Subgraph):
    # First come, first served gradient update
    def build_graph(self):
        def func_fifo_gradient(session, gradients, step_inc, agent_step):
            logger.debug("Gradient with step {} received from agent. Current step: {}"
                         .format(agent_step, session.op_n_step()))
            session.op_apply_gradients(gradients=gradients, increment=step_inc)

        self.func = func_fifo_gradient


class GradientAVG(subgraph.Subgraph):
    # Accumulate gradients from many agents and average them
    def build_graph(self, cfg):
        self.gradients = []
        self.avg_step_inc = 0
        self.cfg = cfg

        def func_average_gradient(session, gradients, step_inc, agent_step):
            logger.debug("Gradient is received, number of gradients collected so far: {}".
                         format(len(self.gradients)))
            if agent_step >= session.op_n_step():
                logger.debug("Gradient is FRESH -> Accepted")
                self.gradients.append(gradients)
                self.avg_step_inc += step_inc
            else:
                logger.debug("Gradient is OLD -> Rejected")

            if len(self.gradients) >= self.cfg.num_gradients:
                # We've collected enough gradients -> we can average them now and make an update step
                logger.debug("Computing Mean of accumulated Gradients")
                flat_grads = [utils.Shaper.get_flat(g) for g in self.gradients]
                mean_flat_grad = np.mean(np.stack(flat_grads), axis=0)

                mean_grad = utils.Shaper.reverse(mean_flat_grad, gradients)
                session.op_apply_gradients(gradients=mean_grad, increment=self.avg_step_inc)
                self.gradients, self.avg_step_inc = [], 0

        self.func = func_average_gradient


class GradientDC(subgraph.Subgraph):
    # Asynchronous Stochastic Gradient Descent with Delay Compensation -> arxiv.org/abs/1609.08326
    def build_graph(self, cfg):
        self.weights_history = {}  # {id: weights}
        self.cfg = cfg
        self.meanSq = .0

        def func_dc_gradient(session, gradients, step_inc, agent_step):
            global_weights_f = session.op_get_weights_flatten()

            old_weights_f = self.weights_history.get(agent_step, global_weights_f)

            new_gradient_f = utils.Shaper.get_flat(gradients)

            # Compute compensated Gradient
            new_gradient_sq = new_gradient_f * new_gradient_f

            if self.cfg.dc_adaptive:
                self.meanSq = self.cfg.dc_m * self.meanSq + (1 - self.cfg.dc_m) * new_gradient_sq
                dc_lambda = self.cfg.dc_lambda / np.sqrt(self.meanSq + self.cfg.dc_epsilon)
            else:
                dc_lambda = self.cfg.dc_lambda

            delta = dc_lambda * (new_gradient_sq * (global_weights_f - old_weights_f))

            compensated_gradient_f = new_gradient_f + delta
            compensated_gradient = utils.Shaper.reverse(compensated_gradient_f, gradients)

            session.op_apply_gradients(gradients=compensated_gradient, increment=step_inc)

            updated_weights = session.op_get_weights_flatten()
            updated_step = session.op_n_step()
            self.weights_history[updated_step] = updated_weights

            # Cleanup history
            for k in list(self.weights_history.keys()):
                if k < updated_step - self.cfg.dc_history:
                    try:
                        del self.weights_history[k]
                    except KeyError:
                        pass

        self.func = func_dc_gradient


class GradientALL(subgraph.Subgraph):
    # Accumulate gradients from many agents and average them
    def build_graph(self, cfg):
        self.gradients = []
        self.weights_history = {}  # {id: weights}
        self.cfg = cfg
        self.avg_step_inc = 0
        self.meanSq = .0

        def func_combine_gradients(session, gradients, step_inc, agent_step):
            logger.debug("Gradient is received, number of gradients collected so far: {}".
                         format(len(self.gradients)))
            gradients_f = utils.Shaper.get_flat(gradients)

            if agent_step >= session.op_n_step():
                logger.debug("Gradient is FRESH -> Accepted")
                self.gradients.append(gradients_f)
                self.avg_step_inc += step_inc
            else:
                global_weights_f = session.op_get_weights_flatten()
                agent_weights_f = self.weights_history.get(agent_step, global_weights_f)
                # Compute compensated Gradient
                gradients_f_sq = gradients_f * gradients_f

                if self.cfg.dc_adaptive:
                    self.meanSq = self.cfg.dc_m * self.meanSq + (1 - self.cfg.dc_m) * gradients_f_sq
                    dc_lambda = self.cfg.dc_lambda / np.sqrt(self.meanSq + self.cfg.dc_epsilon)
                else:
                    dc_lambda = self.cfg.dc_lambda

                delta = dc_lambda * (gradients_f_sq * (global_weights_f - agent_weights_f))
                compensated_gradient_f = gradients_f + delta

                self.gradients.append(compensated_gradient_f)
                logger.debug("Gradient is OLD -> Compensated")

            if len(self.gradients) >= self.cfg.num_gradients:
                # We've collected enough gradients -> we can average them now and make an update step
                logger.debug("Computing Mean of accumulated Gradients")
                mean_flat_grad = np.mean(np.stack(self.gradients), axis=0)

                mean_grad = utils.Shaper.reverse(mean_flat_grad, gradients)
                session.op_apply_gradients(gradients=mean_grad, increment=self.avg_step_inc)
                self.gradients, self.avg_step_inc = [], 0

            updated_weights_f = session.op_get_weights_flatten()
            updated_step = session.op_n_step()
            self.weights_history[updated_step] = updated_weights_f

            # Cleanup history
            for k in list(self.weights_history.keys()):
                if k < updated_step - self.cfg.dc_history:
                    try:
                        del self.weights_history[k]
                    except KeyError:
                        pass

        self.func = func_combine_gradients


class GetVariablesFlatten(subgraph.Subgraph):
    def build_graph(self, variables):
        return tf.concat([tf.reshape(t, [-1]) for t in utils.Utils.flatten(variables.node)], axis=0)


class SetVariablesFlatten(subgraph.Subgraph):
    def build_graph(self, variables):
        self.ph_value = tf.placeholder(tf.float32, [None])
        start = 0
        assignes = []
        for t in utils.Utils.flatten(variables.node):
            shape = t.shape.as_list()
            size = np.prod(shape)
            assignes.append(tf.assign(t, tf.reshape(self.ph_value[start:start + size], shape)))
            start += size
        return tf.group(*assignes)


class DilatedLSTMCell(RNNCell):
    """Dilated LSTM recurrent network cell.
    It is mentioned in FuN algorithm: https://arxiv.org/abs/1703.01161
    """
    def __init__(self, num_units, num_cores, forget_bias=1.0, timestep=0):
        """Initialize the basic LSTM cell.
        Args:
          num_units: int, The number of units in the LSTM cell.
          num_cores: int, The number of partitions (cores) in the LSTM state.
          forget_bias: float, The bias added to forget gates (see above).
        """
        self._num_units = num_units
        self._forget_bias = forget_bias
        # additional variables
        self._cores = tf.constant(num_cores)
        self._timestep = tf.Variable(timestep)  # assign to 0 then terminal (or epoch ends)
        self.reset_timestep = tf.assign(self._timestep, 0)
        # auxiliary operators
        dilated_mask, hold_mask = self._get_mask(num_cores)
        self._dilated_mask = tf.constant(dilated_mask, dtype=tf.float32)
        self._hold_mask = tf.constant(hold_mask, dtype=tf.float32)

    @property
    def state_size(self):
        return 2 * self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Dilated Long short-term memory cell (D-LSTM)."""
        with tf.variable_scope(scope or type(self).__name__):  # "DilatedLSTMCell"
            # Parameters of gates are concatenated into one multiply for efficiency.
            c, h = tf.split(state, 2, axis=1)
            concat = self._linear([inputs, h], 4 * self._num_units, True)

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = tf.split(concat, 4, axis=1)

            new_c = c * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) * tf.tanh(j)
            new_h = tf.tanh(new_c) * tf.sigmoid(o)

            # update relevant cores
            timestep = tf.assign_add(self._timestep, 1)
            core_to_update = tf.mod(timestep, self._cores)

            updated_h = self._hold_mask[core_to_update] * h + self._dilated_mask[core_to_update] * new_h

            return updated_h, tf.concat([new_c, updated_h], axis=1)

    def _linear(self, args, output_size, bias, bias_start=0.0, scope=None):
        """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
        Args:
          args: a 2D Tensor or a list of 2D, batch x n, Tensors.
          output_size: int, second dimension of W[i].
          bias: boolean, whether to add a bias term or not.
          bias_start: starting value to initialize the bias; 0 by default.
          scope: VariableScope for the created subgraph; defaults to "Linear".
        Returns:
          A 2D Tensor with shape [batch x output_size] equal to
          sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
        Raises:
          ValueError: if some of the arguments has unspecified or wrong shape.
        """
        if args is None or (isinstance(args, (list, tuple)) and not args):
            raise ValueError("`args` must be specified")
        if not isinstance(args, (list, tuple)):
            args = [args]

        # Calculate the total size of arguments on dimension 1.
        total_arg_size = 0
        shapes = [a.get_shape().as_list() for a in args]
        for shape in shapes:
            if len(shape) != 2:
                raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
            if not shape[1]:
                raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
            else:
                total_arg_size += shape[1]

        # Computation
        with tf.variable_scope(scope or "Linear"):
            matrix = tf.get_variable("Matrix", [total_arg_size, output_size])
            if len(args) == 1:
                res = tf.matmul(args[0], matrix)
            else:
                res = tf.matmul(tf.concat(args, axis=1), matrix)
            if not bias:
                return res
            bias_term = tf.get_variable(
                "Bias", [output_size],
                initializer=tf.constant_initializer(bias_start))

            # Store as a member for copying (Customized)
            self.matrix = matrix
            self.bias = bias_term

        return res + bias_term

    def _get_mask(self, num_cores):
        basis = np.arange(self._num_units)
        dilated_mask = np.ones(self._num_units)
        hold_mask = np.zeros(self._num_units)
        for i in range(2, num_cores + 1):
            dilated_mask_to_add = (basis % i == 0) * 1
            hold_mask_to_add = dilated_mask[0] - dilated_mask_to_add
            dilated_mask = np.concatenate([dilated_mask, dilated_mask_to_add], axis=0)
            hold_mask = np.concatenate([hold_mask, hold_mask_to_add], axis=0)
        return dilated_mask, hold_mask
