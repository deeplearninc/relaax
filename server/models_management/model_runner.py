import importlib


class ModelRunner(object):

    def __init__(self):
        self.models = {}

    def startModel(self, name, sio, session, namespace='/rlmodels'):
        self.stopModel(session)  # remove old instance of this model
        print("Loading model: " + name)
        module = importlib.import_module("models."+name)
        clazz = getattr(module, self.to_camelcase(name))
        self.models[session] = clazz(sio, session, namespace)
        self.models[session].start()

    def stopModel(self, session):
        if session in self.models:
            print("Removing model for: " + session)
            model = self.models.pop(session)
            print("Stopping model for: " + session)
            model.stop()

    def getModel(self, session):
        return self.models[session]

    @staticmethod
    def to_camelcase(value):
        return ''.join(x.capitalize() or '_' for x in value.split('_'))
