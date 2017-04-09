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


class DilatedBasicLSTMCell(RNNCell):
    """Dilated Basic LSTM recurrent network cell.
    
    Modified to handle groups of sub-states (or 'cores'),
    also as to store matrix and bias as member variable.

    The implementation is based on: http://arxiv.org/abs/1409.2329.

    We add forget_bias (default: 1) to the biases of the forget gate in order to
    reduce the scale of forgetting in the beginning of the training.

    It does not allow cell clipping, a projection layer, and does not
    use peep-hole connections: it is the basic baseline.
    """

    def __init__(self, num_units, cores, forget_bias=1.0, timestep=0):
        """Initialize the basic LSTM cell.

        Args:
          num_units: int, The number of units in the LSTM cell.
          cores: int, The number of sub-states for dilation.
          forget_bias: float, The bias added to forget gates (see above).
          timestep: int, The initial timestep within horizon.
        """
        self._num_units = num_units
        assert cores > 0, 'The number of cores must be greater than zero'
        self._cores = cores
        self._forget_bias = forget_bias
        self.timestep = timestep
        # auxiliary variable for relevant updates
        self._updater = tf.Variable(np.zeros((self._cores, self._num_units)))

    @property
    def state_size(self):
        return (1 + self._cores) * self._num_units

    @property
    def output_size(self):
        return self._cores * self._num_units

    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell (LSTM)."""
        with tf.variable_scope(scope or type(self).__name__):  # "DilatedBasicLSTMCell"
            self.timestep += 1
            # Parameters of gates are concatenated into one multiply for efficiency.
            c, h = tf.split(state, [self._num_units, self.output_size()], axis=1)
            concat = self._linear([inputs, h], 4 * self._num_units, True)

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = tf.split(concat, 4, axis=1)

            new_c = c * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) * tf.tanh(j)
            new_h = tf.tanh(new_c) * tf.sigmoid(o)

            # update only relevant h_i
            h = tf.reshape(h, [self._cores, -1])
            new_h = tf.reshape(new_h, [self._cores, -1])

            self._updater.assign(h)
            idx = self.get_indices()
            new_h = tf.gather(new_h, idx)

            updated_h = tf.scatter_update(self._updater, idx, new_h)
            updated_h = tf.reshape(updated_h, [1, -1])

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

    def get_indices(self):
        indices = []
        for i in range(1, self._cores + 1):
            indices.append(self.timestep % i)
        return list(set(indices))


class DilateBasicLSTMCell(RNNCell):
    """Dilate Basic LSTM recurrent network cell.

    Modified to handle groups of sub-states (or 'cores'),
    also as to store matrix and bias as member variable.

    The implementation is based on: http://arxiv.org/abs/1409.2329.

    We add forget_bias (default: 1) to the biases of the forget gate in order to
    reduce the scale of forgetting in the beginning of the training.

    It does not allow cell clipping, a projection layer, and does not
    use peep-hole connections: it is the basic baseline.
    """

    def __init__(self, num_units, cores, forget_bias=1.0, timestep=0):
        """Initialize the basic LSTM cell.

        Args:
          num_units: int, The number of units in the LSTM cell.
          cores: int, The number of sub-states for dilation.
          forget_bias: float, The bias added to forget gates (see above).
          timestep: int, The initial timestep within horizon.
        """
        self._num_units = num_units
        assert cores > 0, 'The number of cores must be greater than zero'
        self._cores = cores
        self._forget_bias = forget_bias
        self.timestep = timestep
        # auxiliary variable for relevant updates
        self._updater = tf.Variable(np.zeros((self._cores, self._num_units)))

    @property
    def state_size(self):
        return (1 + self._cores) * self._num_units

    @property
    def output_size(self):
        return self._cores * self._num_units

    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell (LSTM)."""
        with tf.variable_scope(scope or type(self).__name__):  # "DilateBasicLSTMCell"
            self.timestep += 1
            # Parameters of gates are concatenated into one multiply for efficiency.
            c, h = tf.split(state, [self._num_units, self.output_size()], axis=1)
            self._updater.assign(h)

            # Get the relevant part of `h` to update
            idx = self.get_indices()

            h_i = tf.reshape(h, [self._cores, -1])
            h_i = tf.gather(h_i, idx)
            h_i = tf.reshape(h_i, [1, -1])

            input_i = tf.reshape(input, [self._cores, -1])
            input_i = tf.gather(input_i, idx)
            input_i = tf.reshape(input_i, [1, -1])

            concat = self._linear([input_i, h_i], 4 * self._num_units, True)

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = tf.split(concat, 4, axis=1)

            new_c = c * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) * tf.tanh(j)
            new_h = tf.tanh(new_c) * tf.sigmoid(o)

            # update only relevant parts of `h`
            repeated_new_h = np.repeat(new_h, len(idx), axis=0)
            updated_h = tf.scatter_update(self._updater, idx, repeated_new_h)
            updated_h = tf.reshape(updated_h, [1, -1])

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
            matrix = tf.get_variable("Matrix", [self._num_units*2, output_size])

            up, down = tf.split(matrix, 2, axis=0)
            times = int(args[0].get_shape().as_list()[1] / self._num_units)
            up_tiled = np.repeat(up, times, axis=0)
            down_tiled = np.repeat(down, times, axis=0)
            new_matrix = tf.concat([up_tiled, down_tiled], axis=0)

            res = tf.matmul(tf.concat(args, axis=1), new_matrix)
            if not bias:
                return res
            bias_term = tf.get_variable(
                "Bias", [output_size],
                initializer=tf.constant_initializer(bias_start))

            # Store as a member for copying (Customized)
            self.matrix = matrix
            self.bias = bias_term

        return res + bias_term

    def get_indices(self):
        indices = []
        for i in range(1, self._cores + 1):
            indices.append(self.timestep % i)
        return list(set(indices))
