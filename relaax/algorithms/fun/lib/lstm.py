from __future__ import print_function

import tensorflow as tf
import numpy as np

from tensorflow.contrib.rnn import RNNCell


class CustomBasicLSTMCell(RNNCell):
    """Custom Basic LSTM recurrent network cell.
    (Modified to store matrix and bias as member variable.)
    The implementation is based on: http://arxiv.org/abs/1409.2329.
    We add forget_bias (default: 1) to the biases of the forget gate in order to
    reduce the scale of forgetting in the beginning of the training.
    It does not allow cell clipping, a projection layer, and does not
    use peep-hole connections: it is the basic baseline.
    For advanced models, please use the full LSTMCell that follows.
    """

    def __init__(self, num_units, forget_bias=1.0):
        """Initialize the basic LSTM cell.
        Args:
          num_units: int, The number of units in the LSTM cell.
          forget_bias: float, The bias added to forget gates (see above).
        """
        self._num_units = num_units
        self._forget_bias = forget_bias

    @property
    def state_size(self):
        return 2 * self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell (LSTM)."""
        with tf.variable_scope(scope or type(self).__name__):  # "BasicLSTMCell"
            # Parameters of gates are concatenated into one multiply for efficiency.
            c, h = tf.split(state, 2, axis=1)
            concat = self._linear([inputs, h], 4 * self._num_units, True)

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = tf.split(concat, 4, axis=1)

            new_c = c * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) * tf.tanh(j)
            new_h = tf.tanh(new_c) * tf.sigmoid(o)

            return new_h, tf.concat([new_c, new_h], axis=1)

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


class DilatedLSTMCell(RNNCell):
    """Dilated LSTM recurrent network cell.
    (Modified to store matrix and bias as member variable.)
    The implementation is based on: http://arxiv.org/abs/1409.2329.
    We add forget_bias (default: 1) to the biases of the forget gate in order to
    reduce the scale of forgetting in the beginning of the training.
    It does not allow cell clipping, a projection layer, and does not
    use peep-hole connections: it is the basic baseline.
    For advanced models, please use the full LSTMCell that follows.
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
        self._timestep = tf.Variable(timestep)  # assign to 0 then terminal (or epoch)
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
        """Long short-term memory cell (LSTM)."""
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
