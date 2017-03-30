from ? import PGConfig


class BaseModel(object):
    def __init__(self):
        self.cfg = PGConfig.preprocess()
        self.build_model()
