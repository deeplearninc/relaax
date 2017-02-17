from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Lambda

from relaax.algorithm_lib.core import *


def make_mlps(config, session):
    K.set_session(session)

    policy_net = Sequential()
    for (i, layeroutsize) in enumerate(config.hidden_layers_sizes):
        input_shape = dict(input_shape=config.state_size) if i == 0 else {}
        policy_net.add(Dense(layeroutsize, activation=config.activation, **input_shape))
    if not config.discrete:
        policy_net.add(Dense(config.action_size))
        policy_net.add(Lambda(lambda x: x * 0.1))
        policy_net.add(ConcatFixedStd())
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


def make_filters(config):
    if config.use_filters:
        obfilter = ZFilter(tuple(config.state_size), clip=5)
        rewfilter = ZFilter((), demean=False, clip=10)
    else:
        obfilter = IDENTITY
        rewfilter = IDENTITY
    return obfilter, rewfilter


def make_wrappers(config, policy_net, value_net, session):

    if not config.discrete:
        probtype = DiagGauss(config.action_size)
    else:
        probtype = Categorical(config.action_size)

    policy = StochPolicyKeras(policy_net, probtype, session)
    baseline = NnVf(value_net, config.timestep_limit, dict(mixfrac=0.1), session)

    return policy, baseline
