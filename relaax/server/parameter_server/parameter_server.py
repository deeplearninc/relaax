import logging
import ruamel.yaml
import time

# Load configuration options
# do it as early as possible
from parameter_server_config import options
from relaax.server.common.bridge.bridge_server import BridgeServer
from relaax.server.common.saver import watch
from relaax.server.common.saver import fs_saver
from relaax.server.common.saver import s3_saver
from relaax.server.common.saver import multi_saver
from relaax.server.common.saver import limited_saver

log = logging.getLogger(__name__)


class ParameterServer(object):

    @staticmethod
    def load_algorithm_ps():
        from relaax.server.common.algorithm_loader import AlgorithmLoader
        try:
            algorithm = AlgorithmLoader.load(options.algorithm_path)
        except Exception:
            log.critical("Can't load algorithm")
            raise

        metrics = None
        return algorithm.ParameterServer(ParameterServer.saver_factory, metrics)

        log.critical("Can't load algorithm's ParameterServer or TFGraph")
        raise

    @staticmethod
    def start():
        try:
            ps = ParameterServer.load_algorithm_ps()

            ps.restore_latest_checkpoint()

            log.info("Staring parameter server server on %s:%d" % options.bind)

            # keep the server or else GC will stop it
            server = BridgeServer(options.bind, ps.session)
            server.start()

            watch = ParameterServer.make_watch(ps)

            while True:
                time.sleep(1)
                watch.check()

        except KeyboardInterrupt:
            # swallow KeyboardInterrupt
            pass
        except:
            raise

    @staticmethod
    def saver_factory(checkpoint):
        ps_options = options.relaax_parameter_server

        savers = []
        if ps_options.checkpoint_dir is not None:
            savers.append(fs_saver.FsSaver(
                checkpoint=checkpoint,
                dir=ps_options.checkpoint_dir
            ))
        if ps_options.checkpoint_aws_s3 is not None:
            aws_access_key, aws_secret_key = ParameterServer.load_aws_keys()
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


def main():
    ParameterServer.start()


if __name__ == '__main__':
    main()
