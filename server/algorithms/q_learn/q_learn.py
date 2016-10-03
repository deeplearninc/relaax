from keras.models import Sequential
from keras.layers.core import Dense, Activation  # Dropout
from keras.optimizers import RMSprop


class QLearn(object):
    def __init__(self, inputLen, outputLen):
        self.model = Sequential()
        self.model.add(Dense(164, init='lecun_uniform', input_shape=(inputLen,)))
        self.model.add(Activation('relu'))
        # self.model.add(Dropout(0.2)) I'm not using dropout, but maybe you wanna give it a try?

        self.model.add(Dense(150, init='lecun_uniform'))
        self.model.add(Activation('relu'))
        # self.model.add(Dropout(0.2))

        self.model.add(Dense(outputLen, init='lecun_uniform'))
        self.model.add(Activation('linear'))  # linear output so we can have range of real-valued outputs

        rms = RMSprop()
        self.model.compile(loss='mse', optimizer=rms)
        print(self.model)

    def predict(self, state, batch_size):
        return self.model.predict(state, batch_size)

    def fit(self, x, y, batch_size, nb_epoch, verbose):
        self.model.fit(x, y, batch_size, nb_epoch, verbose)
