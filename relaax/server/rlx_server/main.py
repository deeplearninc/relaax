import os
import sys

sys.path.append(os.path.normpath(os.path.join(
    os.path.dirname(__file__),
    '../../..'
)))

import relaax.server.rlx_server

relaax.server.rlx_server.main()
