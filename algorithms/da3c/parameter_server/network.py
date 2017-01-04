from ..common import network


def make(config, thread_index):
    return network.make_shared_network(config, thread_index)
