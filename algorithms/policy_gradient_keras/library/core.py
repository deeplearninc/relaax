from .config import *   # PGConfig
from .loss import *
from .reward import *


def initialize():
    session = tf.Session()
    keras.backend.set_session(session)
    session.run(tf.variables_initializer(tf.global_variables()))
    return session


# === Session dependent functions below ===

# === Move below to some appropriate library parts ===

