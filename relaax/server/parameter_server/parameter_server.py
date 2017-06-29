from __future__ import absolute_import
from __future__ import division
from builtins import object
import logging
import ruamel.yaml
import time
import threading
import signal

# Load configuration options
# do it as early as possible
from .parameter_server_config import options
from relaax.server.common.bridge import bridge_server
from relaax.server.common.metrics import tensorflow_metrics
from relaax.server.common.saver import fs_saver
from relaax.server.common.saver import limited_saver
from relaax.server.common.saver import multi_saver
from relaax.server.common.saver import s3_saver
from relaax.server.common.saver import watch
import multiprocessing
try:
    from Queue import Queue, Empty  # noqa
except ImportError:
    from queue import Queue, Empty  # noqa
    
log = logging.getLogger(__name__)

class ParameterServer(object):

    @classmethod
    def load_algorithm_ps(cls):
        from relaax.server.common.algorithm_loader import AlgorithmLoader
        try:
            ParameterServer = AlgorithmLoader.load_parameter_server(
                    options.algorithm_path, options.algorithm_name)
        except Exception:
            log.critical("Can't load algorithm")
            raise

        return ParameterServer(cls.saver_factory, cls.metrics_factory)

    @classmethod    
    def exit_server(cls, signum, frame):
        cls.stopped_server = True
            
    @classmethod
    def start(cls):
        try:
            log.info("Starting parameter server on %s:%d" % options.bind)

            init_ps = CallOnce(cls.init)

            # keep the server or else GC will stop it
            server = bridge_server.BridgeServer(options.bind, init_ps)
            server.start()

            ps = init_ps()
            watch = cls.make_watch(ps)

            speedm = Speedometer(ps)
            events = multiprocessing.Queue()
            signal.signal(signal.SIGINT, cls.exit_server)
            signal.signal(signal.SIGTERM, cls.exit_server)
            cls.stopped_server = False

            while not cls.stopped_server:
                #time.sleep(1)
                try:
                    msg = events.get(timeout=1)
                except Empty:
                    pass
                except:
                    break    
                else:
                    watch.check()
            
            ps.save_checkpoint()
            speedm.stop_timer()
        except KeyboardInterrupt:
            # swallow KeyboardInterrupt
            pass
        except:
            raise

    @classmethod
    def init(cls):
        ps = cls.load_algorithm_ps()
        ps.restore_latest_checkpoint()
        return ps

    @classmethod
    def saver_factory(cls, checkpoint):
        ps_options = options.relaax_parameter_server

        savers = []
        if ps_options.checkpoint_dir is not None:
            savers.append(fs_saver.FsSaver(
                checkpoint=checkpoint,
                dir=ps_options.checkpoint_dir
            ))
        if ps_options.checkpoint_aws_s3 is not None:
            aws_access_key, aws_secret_key = cls.load_aws_keys()
            savers.append(s3_saver.S3Saver(
                checkpoint=checkpoint,
                bucket_key=ps_options.checkpoint_aws_s3,
                aws_access_key=aws_access_key,
                aws_secret_key=aws_secret_key
            ))

        saver = multi_saver.MultiSaver(savers)
        if ps_options.checkpoints_to_keep is not None:
            saver = limited_saver.LimitedSaver(
                saver,
                ps_options.checkpoints_to_keep
            )
        return saver

    @staticmethod
    def metrics_factory(x):
        return tensorflow_metrics.TensorflowMetrics(options.relaax_parameter_server.metrics_dir, x)

    @staticmethod
    def load_aws_keys():
        aws_keys = options.relaax_parameter_server.aws_keys
        if aws_keys is None:
            return None, None
        with open(aws_keys, 'r') as f:
            aws_keys = ruamel.yaml.load(f, Loader=ruamel.yaml.Loader)
        return aws_keys['access'], aws_keys['secret']

    @staticmethod
    def make_watch(ps):
        return watch.Watch(ps,
            (
                options.relaax_parameter_server.checkpoint_step_interval,
                lambda: ps.n_step()
            ), (
                options.relaax_parameter_server.checkpoint_time_interval,
                lambda: time.time()
            )
        )


class CallOnce(object):
    def __init__(self, f):
        self.lock = threading.Lock()
        self.f = f
        self.initialized = False

    def __call__(self):
        with self.lock:
            if not self.initialized:
                self.cache = self.f()
                self.initialized = True
        return self.cache


class Speedometer(object):
    def __init__(self, ps):
        self.ps = ps
        self.run_timer(time.time(), ps.session.op_n_step())

    def measure(self, start_time, start_n_step):
        current_time = time.time()
        current_n_step = self.ps.session.op_n_step()
        self.ps.metrics.scalar('steps_per_sec', (current_n_step - start_n_step) / (current_time - start_time))
        self.run_timer(current_time, current_n_step)

    def run_timer(self, start_time, start_n_steps):
        self.timer=threading.Timer(60, self.measure, args=(start_time, start_n_steps))
        self.timer.start()
        
    def stop_timer(self):
        self.timer.cancel()

def main():
    ParameterServer.start()

if __name__ == '__main__':
    main()
