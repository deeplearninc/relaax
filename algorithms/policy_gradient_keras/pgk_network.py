from library.core import *


# Policy Neural Network
class PolicyNN(object):
    def __init__(self):
        self.cfg = PGConfig.preprocess()

        # model's network creation
        input_ = Input(shape=(self.cfg.state_size,))
        throw_ = FullyConnected(self.cfg.layers_size[0], activation='elu',
                                init='glorot_uniform')(input_)

        for i in range(1, len(self.cfg.hidden_layers)):
            throw_ = FullyConnected(self.cfg.layers_size[i], activation='elu',
                                    init='glorot_uniform')(throw_)

        throw_ = FullyConnected(self.cfg.action_size, activation='softmax',
                                init='glorot_uniform')(throw_)
        self.net = Model(input=input_, output=throw_)

    @define_scope
    def compute_gradients(self, loss):
        return tf.gradients(loss.eval, self.net.trainable_weights)

    @define_scope
    def apply_gradients(self, optimizer_name):
        self.gradients =\
            [tf.placeholder(v.dtype, v.get_shape()) for v in self.net.trainable_weights]

        if optimizer_name == 'Adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=self.cfg.learning_rate)
        else:
            assert False

        return optimizer.apply_gradients(zip(self.gradients, self.net.trainable_weights))


class AgentModel(PolicyNN):
    def __init__(self):
        super(AgentModel, self).__init__()
        self.compute_gradients(SimpleLoss(self.net.output))


class GlobalModel(PolicyNN):
    def __init__(self):
        super(GlobalModel, self).__init__()
        self.apply_gradients('Adam')
