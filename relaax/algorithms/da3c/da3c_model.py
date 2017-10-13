from __future__ import absolute_import

from relaax.common.algorithms.lib import utils

from . import da3c_config
from . import da3c_discrete_model
from . import da3c_continuous_model


if da3c_config.config.output.continuous:
    SharedParameters = da3c_continuous_model.SharedParameters
    AgentModel = da3c_continuous_model.AgentModel
else:
    SharedParameters = da3c_discrete_model.SharedParameters
    AgentModel = da3c_discrete_model.AgentModel

if __name__ == '__main__':
    utils.assemble_and_show_graphs(SharedParameters, AgentModel)
