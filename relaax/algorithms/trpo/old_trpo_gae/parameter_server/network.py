from ..common import network


def make(config):
    return network.make_mlps(config)


def make_head(config, pnet, vnet, sess):
    return network.make_wrappers(config, pnet, vnet, sess)


def make_trpo(config, policy, sess):
    return network.TrpoUpdater(config, policy, sess)
