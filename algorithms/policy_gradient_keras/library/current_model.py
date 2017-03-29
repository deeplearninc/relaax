
model = Model()

def reset_model(model):
    model = DummyModel()


class DummyModel(object):
    def __init__(self):
        self.last_layer = DummyLayer()
        reset_model(self)

    def last_layer(self):
        return self.last_layer

    def add_layer(self, layer):
        pass


class SequentialModel(object):
    def __init__(self):
        self.last_layer = DummyLayer()
        reset_model(self)

    def last_layer(self):
        return self.last_layer

    def add_layer(self, layer):
        self.last_layer = layer


class GraphModel(object):
    def __init__(self):
        self.last_layer = DummyLayer()
        reset_model(self)

    def last_layer(self):
        return self.last_layer

    def add_layer(self, layer):
        pass
