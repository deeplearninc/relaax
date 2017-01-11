

class BridgeBase(object):
    def increment_global_t(self):
        raise NotImplementedError

    def apply_gradients(self, gradients):
        raise NotImplementedError

    def get_values(self):
        raise NotImplementedError

    def metrics(self):
        raise NotImplementedError


class BridgeControlBase(object):
    def parameter_server_stub(self, parameter_server_url):
        raise NotImplementedError

    def start_parameter_server(self, address, service):
        raise NotImplementedError
