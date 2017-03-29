from library.current_model import model
from library.utils import ComputeGradients, ApplyGradients
from library.optimizers import Adam


class BaseModel(object):
    def __init__(self):
        self.cfg = PGConfig.preprocess()
        self.build_model()



class AgentModel(object):
    def build_model(self):

        SequentialModel()
        Input(self.cfg.state_size,)
        for layer in self.cfg.hidden_layers:
            FullyConnected(layer, activation='elu', init='glorot_uniform')
        FullyConnected(self.cfg.action_size, activation='softmax', init='glorot_uniform')

        model.loss = SimpleLoss()
        self.compute_gradients = ComputeGradients(model.loss)

    @property
    def compute_gradients(self):
        return self.model.compute_gradients


def parameter_server_model():
    cfg = PGConfig.preprocess()

    model = DummyModel()
    for layer in cfg.hidden_layers:
        Weights(layer)

    model.apply_gradients = ApplyGradients(Adam())

    return model
