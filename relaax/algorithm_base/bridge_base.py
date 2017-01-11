

class BridgeControlBase(object):
    def parameter_server_stub(self, parameter_server_url):
        raise NotImplementedError

    def start_parameter_server(self, address, service):
        raise NotImplementedError
