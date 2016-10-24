import sys
sys.path.append('../../server')

import time
import signal
import tensorflow as tf
import algorithms.a3c.params
import os

import algorithms.a3c.game_ac_network
import shared


def main():

    signal.signal(signal.SIGINT, lambda _1, _2: sys.exit(0))

    server = shared.ps()

    params = algorithms.a3c.params.Params()

    kernel = "/cpu:0"
    if params.use_GPU:
        kernel = "/gpu:0"

    with tf.device(shared.ps_device() + kernel):
        global_network = algorithms.a3c.game_ac_network.make_shared_network(params, -1)

    initialize = tf.initialize_all_variables()

    sess = tf.Session(
        target=server.target,
        config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    )

    sess.run(initialize)

    lstm_str = ''
    if params.use_LSTM:
        lstm_str = 'lstm_'
    checkpoint_dir = 'checkpoints/' + 'boxing' + '_a3c_' + \
                          lstm_str + str(params.threads_cnt) + 'threads'

    # init or load checkpoint with saver
    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("checkpoint loaded:", checkpoint.model_checkpoint_path)
        tokens = checkpoint.model_checkpoint_path.split("-")
        # set global step
        global_t = int(tokens[1])
        print(">>> global step set: ", global_t)
    else:
        global_t = 0
        print("Could not find old checkpoint")

    def stop_server(_1, _2):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        # TODO: very bad global_t increment. See the same variable in trainer.py
        saver.save(sess, checkpoint_dir + '/' + 'checkpoint', global_step=global_t + 1)
        sess.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, stop_server)

    last_global_t = None
    while True:
        time.sleep(1)
        global_t = sess.run(global_network.global_t)
        if global_t != last_global_t:
            last_global_t = global_t
            print("global_t is %d" % global_t)


if __name__ == '__main__':
    main()
