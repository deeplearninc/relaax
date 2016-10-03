# from misc_utils import comma_sep_ints
from algos import *     # comma_sep_ints(misc_utils), PG_OPTIONS(core)
from algos.trpo import TrpoUpdater
from gym.spaces import Box, Discrete

# NEED Session THERE !!!
from keras.models import Sequential
from keras.layers.core import Dense, Lambda, TimeDistributedDense

from keras.layers.recurrent import LSTM
# from keras.layers.wrappers import TimeDistributed

from keras import backend as K
import os


MLP_OPTIONS = [
    ("hid_sizes", comma_sep_ints, [64, 64], "Sizes of hidden layers of MLP"),
    ("activation", str, "tanh", "nonlinearity")
]


def make_mlps(ob_space, ac_space, cfg, session):
    K.set_session(session)
    assert isinstance(ob_space, Box)
    hid_sizes = cfg["hid_sizes"]
    if isinstance(ac_space, Box):
        outdim = ac_space.shape[0]
        probtype = DiagGauss(outdim)
    elif isinstance(ac_space, Discrete):
        outdim = ac_space.n
        probtype = Categorical(outdim)

    net = Sequential()
    '''
    for (i, layeroutsize) in enumerate(hid_sizes):
        inshp = dict(input_shape=ob_space.shape) if i == 0 else {}
        net.add(Dense(layeroutsize, activation=cfg["activation"], **inshp))
    '''
    net.add(TimeDistributedDense(64, input_shape=(ob_space.shape[0], 4), activation=cfg["activation"]))
    net.add(TimeDistributedDense(64, activation=cfg["activation"]))     # input_shape=(4, ob_space.shape[0])
    # '''
    if isinstance(ac_space, Box):
        net.add(LSTM(64, unroll=True))     # new --> add LSTM     return_sequences=True
        net.add(Dense(outdim))
        net.add(Lambda(lambda x: x * 0.1))
        # Wlast = net.layers[-1].W
        # Wlast.set_value(Wlast.get_value(borrow=True)*0.1)
        net.add(ConcatFixedStd())
    else:
        net.add(Dense(outdim, activation="softmax"))
        net.add(Lambda(lambda x: x * 0.1))
        # Wlast = net.layers[-1].W
        # Wlast.set_value(Wlast.get_value(borrow=True)*0.1)
    policy = StochPolicyKeras(net, probtype, session)   # SESSION !!! --> need ADD
    vfnet = Sequential()
    '''
    for (i, layeroutsize) in enumerate(hid_sizes):
        inshp = dict(input_shape=(ob_space.shape[0]+1,)) if i == 0 else {}  # add one extra feature for timestep
        vfnet.add(Dense(layeroutsize, activation=cfg["activation"], **inshp))
    '''
    vfnet.add(TimeDistributedDense(64, activation=cfg["activation"], input_shape=(ob_space.shape[0], 4)))   # +1
    vfnet.add(TimeDistributedDense(64, activation=cfg["activation"]))
    vfnet.add(LSTM(64, unroll=True))     # new --> add LSTM     return_sequences=True
    # '''
    vfnet.add(Dense(1))
    baseline = NnVf(vfnet, session, cfg["timestep_limit"], dict(mixfrac=0.1))   # SESSION !!! --> need ADD
    return policy, baseline, net, vfnet     # add return for Keras nets for saving


FILTER_OPTIONS = [
    ("filter", int, 1, "Whether to do a running average filter of the incoming observations and rewards")
]


def make_filters(cfg, ob_space):
    if cfg["filter"]:
        obfilter = ZFilter(ob_space.shape, clip=5)
        rewfilter = ZFilter((), demean=False, clip=10)
    else:
        obfilter = IDENTITY
        rewfilter = IDENTITY
    return obfilter, rewfilter


class AgentWithPolicy(object):
    def __init__(self, policy, obfilter, rewfilter):
        self.policy = policy
        self.obfilter = obfilter
        self.rewfilter = rewfilter
        self.stochastic = True

    def set_stochastic(self, stochastic):
        self.stochastic = stochastic

    def act(self, ob_no):
        return self.policy.act(ob_no, stochastic=self.stochastic)

    def get_flat(self):
        return self.policy.get_flat()

    def set_from_flat(self, th):
        return self.policy.set_from_flat(th)

    def obfilt(self, ob):
        return self.obfilter(ob)

    def rewfilt(self, rew):
        return self.rewfilter(rew)


class TrpoAgent(AgentWithPolicy):
    options = MLP_OPTIONS + PG_OPTIONS + TrpoUpdater.options + FILTER_OPTIONS

    def __init__(self, ob_space, ac_space, usercfg, session):   # SESSION !!! --> need ADD
        algo_name = '_trpo_'
        misc = 'lstm'
        self.CHECKPOINT_DIR = 'checkpoints/' + usercfg["env"] + algo_name + misc

        cfg = update_default_config(self.options, usercfg)
        policy, self.baseline, self.pnet, self.vfnet \
            = make_mlps(ob_space, ac_space, cfg, session)   # SESSION !!! --> need ADD
        obfilter, rewfilter = make_filters(cfg, ob_space)
        self.updater = TrpoUpdater(policy, cfg, session)    # SESSION !!! --> need ADD
        AgentWithPolicy.__init__(self, policy, obfilter, rewfilter)

        session.run(tf.initialize_all_variables())

    def save(self, n_iter):
        if not os.path.exists(self.CHECKPOINT_DIR):
            os.makedirs(self.CHECKPOINT_DIR)
        self.pnet.save_weights(self.CHECKPOINT_DIR+"/pnet--"+str(n_iter)+".h5")
        self.vfnet.save_weights(self.CHECKPOINT_DIR+"/vfnet--"+str(n_iter)+".h5")

    def restore(self):
        n_iter = 0
        if os.path.exists(self.CHECKPOINT_DIR):
            for filename in os.listdir(self.CHECKPOINT_DIR):
                tokens = filename.split("--")
                n_iter = int(tokens[1].split(".")[0])
                if tokens[0] == 'pnet':
                    self.pnet.load_weights(self.CHECKPOINT_DIR+"/pnet--"+str(n_iter)+".h5")
                else:
                    self.vfnet.load_weights(self.CHECKPOINT_DIR+"/vfnet--"+str(n_iter)+".h5")
        return n_iter
