from library.core import *


# Policy Neural Network
class PolicyNN(object):
    def __init__(self):
        cfg = PGConfig.preprocess()
        self.lr = cfg.learning_rate

        # model's network creation
        input_ = Input(shape=(cfg.state_size,))
        throw_ = FullyConnected(cfg.layers_size[0], activation='elu',
                                init='glorot_uniform')(input_)

        for i in range(1, len(cfg.hidden_layers)):
            throw_ = FullyConnected(cfg.layers_size[i], activation='elu',
                                    init='glorot_uniform')(throw_)

        throw_ = FullyConnected(cfg.action_size, activation='softmax',
                                init='glorot_uniform')(throw_)
        self.net = Model(input=input_, output=throw_)


class AgentNN(PolicyNN):
    def __init__(self):
        super(AgentNN, self).__init__()
        self.loss = SimpleLoss(self.net.output)             # just op --> created new labels
        self.compute_gradients = compute_gradients(self)    # just op


class GlobalNN(PolicyNN):
    def __init__(self):
        super(GlobalNN, self).__init__()
        self.gradients = None   # init below --> label, which needs to feed
        self.apply = apply_gradients(self, 'Adam')  # just op
