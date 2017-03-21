import argparse
import ruamel.yaml


def load():
    parser = argparse.ArgumentParser()

    # Define command line arguments
    parser.add_argument('--config', type=str, default=None,
                        help='RELAAX config yaml file')
    parser.add_argument('--ws-address', type=str, default=None,
                        help=('address of the Web Sockets server (host:port); '
                              'in config yaml it is --address option '
                              'in wsproxy section'))
    parser.add_argument('--rlx-address', type=str, default=None,
                        help=('address of the RLX server (host:port); '
                              'in config yaml it is --bind option in '
                              'relaax-rlx-server section'))

    # Parse command line arguments
    args = parser.parse_args()

    # If parameters are not specified in command
    # line try read them from config yaml
    if args.config:

        with open(args.config, 'r') as f:
            yaml = ruamel.yaml.load(f, Loader=ruamel.yaml.Loader)

            if 'relaax-rlx-server' in yaml:
                options = yaml['relaax-rlx-server']
                if args.rlx_address is None and '--bind' in options:
                    args.rlx_address = options['--bind']

            if 'wsproxy' in yaml:
                options = yaml['wsproxy']
                if args.ws_address is None and '--address' in options:
                    args.ws_address = options['--address']

    # If parameters are not set in command
    # line or config, assume defaults
    if args.rlx_address is None:
        args.rlx_address = "localhost:7001"
    if args.ws_address is None:
        args.ws_address = "localhost:9000"

    # Simple check of the server addresses format
    args.rlx_address = map(lambda x: x.strip(), args.rlx_address.split(':'))
    if len(args.rlx_address) != 2:
        print "Error! Please specify RLX server address in host:port format"
        exit()

    args.ws_address = map(lambda x: x.strip(), args.ws_address.split(':'))
    if len(args.ws_address) != 2:
        print('Error! Please specify Web Sockets ',
              'server address in host:port format')
        exit()

    return args


options = load()
