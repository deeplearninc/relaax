import importlib


class AleModel(object):
    def __init__(self, params, Trainer, log_dir):
        self._params = params
        self._Trainer = Trainer
        self._log_dir = log_dir
        self._trainer = None        # assign via init_model event in init_model method

    def threads_cnt(self):
        return self._params.threads_cnt

    def init_model(self, ps_stub):  # init model's algorithm with the given parameters
        self._trainer = self._Trainer(
            self._params,
            ps_stub=ps_stub,
            log_dir=self._log_dir
        )

    def getAction(self, message):
        return self._trainer.getAction(message)

    def addEpisode(self, message):
        return self._trainer.addEpisode(message)
