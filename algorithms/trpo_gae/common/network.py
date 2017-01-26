from keras.models import Sequential
from keras.layers.core import Dense, Lambda

from . import core


def make_mlps(config):

    policy_net = Sequential()
    for (i, layeroutsize) in enumerate(config.hidden_layers_sizes):
        input_shape = dict(input_shape=config.state_size) if i == 0 else {}
        policy_net.add(Dense(layeroutsize, activation=config.activation, **input_shape))
    if not config.discrete:
        policy_net.add(Dense(config.action_size))
        policy_net.add(Lambda(lambda x: x * 0.1))
        policy_net.add(core.ConcatFixedStd())
    else:
        policy_net.add(Dense(config.action_size, activation="softmax"))
        policy_net.add(Lambda(lambda x: x * 0.1))

    value_net = Sequential()
    for (i, layeroutsize) in enumerate(config.hidden_layers_sizes):
        # add one extra feature for timestep
        input_shape = dict(input_shape=(config.state_size[0]+1,)) if i == 0 else {}
        value_net.add(Dense(layeroutsize, activation=config.activation, **input_shape))
    value_net.add(Dense(1))

    return policy_net, value_net


def make_wrappers(config, policy_net, value_net, session):

    if not config.discrete:
        probtype = core.DiagGauss(config.action_size)
    else:
        probtype = core.Categorical(config.action_size)
