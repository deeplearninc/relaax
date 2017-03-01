from ..common import network


def make(config, sess):
    return network.make_mlps(config, sess)


def make_filters(config):
    return network.make_filters(config)


def make_head(config, pnet, vnet, sess):
    return network.make_wrappers(config, pnet, vnet, sess)
