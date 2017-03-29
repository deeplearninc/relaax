from library.current_model import model
from library.utils import ComputeGradients, ApplyGradients
from library.optimizers import Adam


class AgentModel():
    def __init__(self):
        cfg = PGConfig.preprocess()

        SequentialModel()
        Input(cfg.state_size,)
        for layer in cfg.hidden_layers:
            FullyConnected(layer, activation='elu', init='glorot_uniform')
        FullyConnected(cfg.action_size, activation='softmax', init='glorot_uniform')

        model.loss = SimpleLoss()
        model.compute_gradients = ComputeGradients(model.loss)

    @property
    def compute_gradients(self):
        return model.compute_gradients


def parameter_server_model():
    cfg = PGConfig.preprocess()

    model = DummyModel()
    for layer in cfg.hidden_layers:
        Weights(layer)

    model.apply_gradients = ApplyGradients(Adam())

    return model
