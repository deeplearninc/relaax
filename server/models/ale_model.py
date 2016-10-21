from base_model import BaseModel
import importlib
import json


class AleModel(BaseModel):
    def __init__(self, *args, **kwargs):
        super(AleModel, self).__init__(*args, **kwargs)
        self.algo_name = None   # assign via get_params event in init_params method
        self.params = None      # assign via get_params event in init_params method
        self.algo = None        # assign via init_model event in init_model method

    def init_params(self, algo_name):
        module = importlib.import_module("algorithms." + algo_name + ".params")
        clazz = getattr(module, 'Params')
        self.params = clazz()  # get the instance of Params Class to perform
        self.algo_name = algo_name
        self.sio.emit('init params', json.dumps({'threads_cnt': self.params.threads_cnt}),
                      room=self.session, namespace=self.namespace)

    def init_model(self, message, target='', global_device='', local_device=''):  # init model's algorithm with the given parameters
        print(message)

        module = importlib.import_module("algorithms." + self.algo_name + ".trainer")
        clazz = getattr(module, 'Trainer')
        self.algo = clazz(self.params, target=target, global_device=global_device, local_device=local_device)

        self.sio.emit(
            'model is ready',
            {'threads_cnt': self.params.threads_cnt},
            room=self.session,
            namespace=self.namespace
        )

    def getAction(self, message):
        return self.algo.getAction(message)

    def addEpisode(self, message):
        self.sio.emit('episode ack', json.dumps(self.algo.addEpisode(message)),
                      room=self.session, namespace=self.namespace)
