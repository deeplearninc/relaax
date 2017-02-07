from relaax.server.common.saver.saver import Saver
from os import path, makedirs
from json import dump   # cPickle | ujson


class KerasSaver(Saver):
    def __init__(self, directory):
        super(KerasSaver, self).__init__()
        self.dir = directory

    def latest_checkpoint_idx(self):
        latest_cp_path = self.dir + '/latest'
        if path.isfile(latest_cp_path):
            with open(latest_cp_path, 'r') as f:
                net_idx = f.readline()
                data_idx = f.readline()
            return net_idx, data_idx    # True
        return 0, 0                     # False

    def save_checkpoint(self, pnet, vnet, n_iter, data, length):
        if not path.exists(self.dir):
            makedirs(self.dir)
        pnet.save_weights(self.dir + "/pnet--" + str(n_iter) + ".h5")
        vnet.save_weights(self.dir + "/vnet--" + str(n_iter) + ".h5")
        with open(self.dir + "/data--" + str(length) + ".json", 'wb') as datafile:
            dump(data, datafile)

    def location(self):
        return "'%s' dir" % self.dir
