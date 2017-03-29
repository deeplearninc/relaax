from library.current_model import model
from library.utils import ComputeGradients, ApplyGradients
from library.optimizers import Adam


def agent_model():
    cfg = PGConfig.preprocess()

    SequentialModel()
    Input(cfg.state_size,)
    for layer in cfg.hidden_layers:
        FullyConnected(layer, activation='elu', init='glorot_uniform')
    FullyConnected(cfg.action_size, activation='softmax', init='glorot_uniform')

    model.loss = SimpleLoss()
    model.compute_gradients = ComputeGradients(model.loss)

    return model


def parameter_server_model():
    cfg = PGConfig.preprocess()

    model = DummyModel()
    for layer in cfg.hidden_layers:
        Weights(layer)

    model.apply_gradients = ApplyGradients(Adam())

    return model
