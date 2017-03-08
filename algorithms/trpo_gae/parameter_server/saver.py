from relaax.server.common.saver.saver import Saver
from os import path, makedirs
from cPickle import dump    # ujson


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
                global_idx = f.readline()
            return True, int(net_idx), int(data_idx), int(global_idx)
        return False, 0, 0, 0

    def save_checkpoint(self, pnet, vnet, n_iter, data, length, g_step):
        if not path.exists(self.dir):
            makedirs(self.dir)
        pnet.save_weights(self.dir + "/pnet--" + str(n_iter) + ".h5")
        vnet.save_weights(self.dir + "/vnet--" + str(n_iter) + ".h5")
        with open(self.dir + "/data--" + str(n_iter) + "-" + str(length) + ".p", 'wb') as datafile:
            dump(data, datafile)
        with open(self.dir + '/latest', 'w') as f:
            f.write(str(n_iter) + '\n' + str(length) + '\n' + str(g_step))

    def location(self):
        return "'%s' dir" % self.dir
