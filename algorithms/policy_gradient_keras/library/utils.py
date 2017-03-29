from current_model import model


def compute_gradients(loss):
    return tf.gradients(loss, model.trainable_weights)

def apply_gradients(optimizer):
    model.gradients = [
        tf.placeholder(v.dtype, v.get_shape()) for v in model.trainable_weights
    ]
    model.optimizer = optimizer 
