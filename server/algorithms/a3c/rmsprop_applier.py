import tensorflow as tf
from tensorflow.python.training import training_ops
from tensorflow.python.training import slot_creator


class _Slots(object):
    RMS = 'rms'
    MOMENTUM = 'momentum'

    def __init__(self, name):
        self._name = name
        self._slots = {}

    def create_rms(self, var):
        self._make_slot(
            var,
            self.RMS,
            lambda: slot_creator.create_slot(
                var,
                tf.constant(1.0, dtype=var.dtype, shape=var.get_shape()),
                self._name
            )
        )

    def create_momentum(self, var):
        self._make_slot(
            var,
            self.MOMENTUM,
            lambda: slot_creator.create_zeros_slot(var, self._name)
        )

    def get_rms(self, var):
        return self._get_slot(var, self.RMS)

    def get_momentum(self, var):
        return self._get_slot(var, self.MOMENTUM)

    def _make_slot(self, var, slot_name, factory):
        key = (var, slot_name)
        if key in self._slots:
            return
        self._slots[key] = factory()

    def _get_slot(self, var, slot_name):
        return self._slots.get((var, slot_name))


class RMSPropApplier(object):
    def __init__(self,
                 learning_rate,
                 decay=0.9,
                 momentum=0.0,
                 epsilon=1e-10,
                 clip_norm=40.0,
                 global_vars=[],
                 device="/cpu:0",
                 name="RMSPropApplier"):

        self._name = name
        self._learning_rate = learning_rate
        self._decay = decay
        self._momentum = momentum
        self._epsilon = epsilon
        self._clip_norm = clip_norm
        self._global_vars = global_vars
        self._device = device

        self._slots = _Slots(name)

        with tf.device(device):
            for var in global_vars:
                self._slots.create_rms(var)
                self._slots.create_momentum(var)

    # Apply accumulated gradients to var.
    def __call__(self, accum_grad_list):
        with tf.device(self._device):
            with tf.name_scope(None, self._name, []) as name:
                # Tensors for learning rate and momentum.
                learning_rate = tf.convert_to_tensor(self._learning_rate, name="learning_rate")
                decay         = tf.convert_to_tensor(self._decay        , name="decay"        )
                momentum      = tf.convert_to_tensor(self._momentum     , name="momentum"     )
                epsilon       = tf.convert_to_tensor(self._epsilon      , name="epsilon"      )

                update_ops = []

                for var, accum_grad in zip(self._global_vars, accum_grad_list):
                    with tf.name_scope("update_" + var.op.name), tf.device(var.device):
                        update_ops.append(
                            # TODO: in RMSProp native code, memcpy() (for CPU) and
                            # cudaMemcpyAsync() (for GPU) are used when updating values,
                            # and values might tend to be overwritten with results from other threads.
                            # (Need to check the learning performance with replacing it)
                            training_ops.apply_rms_prop(
                                var,
                                self._slots.get_rms(var),
                                self._slots.get_momentum(var),
                                learning_rate,
                                decay,
                                momentum,
                                epsilon,
                                tf.clip_by_norm(accum_grad, self._clip_norm),
                                use_locking=False
                            ).op
                        )

                return tf.group(*update_ops, name=name)
