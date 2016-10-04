class ServerAPI:
    def on_session_id(self, *args):
        print('on_session_response', args)
        self.emit('join', {'room': args[0]['session_id']})
