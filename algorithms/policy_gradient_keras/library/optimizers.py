from current_model import model

def adam(learning_rate):
    return tf.train.AdamOptimizer(learning_rate=learning_rate, gradients=model.gradients)
