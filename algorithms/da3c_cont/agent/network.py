from ..common import network

def make(config, thread_index):
    return network.make_full_network(config, thread_index)

