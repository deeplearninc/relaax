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


class AgentModel(PolicyNN):
    def __init__(self):
        super(AgentModel, self).__init__()
        self.loss = SimpleLoss(self.net.output)             # just op --> created new labels
        self.compute_gradients = compute_gradients(self)    # just op


class GlobalModel(PolicyNN):
    def __init__(self):
        super(GlobalModel, self).__init__()
        self.gradients = None   # init below --> label, which needs to feed
        self.apply_gradients = apply_gradients(self, 'Adam')  # just op
