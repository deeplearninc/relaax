import importlib


class AleModel(object):
    def __init__(self, params, Trainer):
        self._params = params
        self._Trainer = Trainer
        self._trainer = None        # assign via init_model event in init_model method

    def threads_cnt(self):
        return self._params.threads_cnt

    def init_model(self, target='', global_device='', local_device=''):  # init model's algorithm with the given parameters
        self._trainer = self._Trainer(
            self._params,
            target=target,
            global_device=global_device,
            local_device=local_device
        )

    def getAction(self, message):
        return self._trainer.getAction(message)

    def addEpisode(self, message):
        return self._trainer.addEpisode(message)
