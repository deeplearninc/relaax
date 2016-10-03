import time
from base_model import BaseModel


class SomeModel(BaseModel):
    def __init__(self, *args, **kwargs):
        super(SomeModel, self).__init__(*args, **kwargs)

    def train(self):
        while not self.isStopped():
            time.sleep(10)
            print("Sending message to room: " + self.session)
            self.sio.emit('model message', {'data': 'looping...'}, room=self.session, namespace=self.namespace)
