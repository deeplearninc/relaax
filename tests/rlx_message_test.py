from builtins import object
import numpy as np
import numpy.testing as npt

from relaax.common.rlx_message import RLXMessage


class TestRLXMessage(object):

    def test_to_wire_and_back(self):
        npar = np.array([1.001, 2.002, 3.003])
        data = {'arg1': 1, 'array': [4., 5., 6.], 'nparray': npar}
        wire = RLXMessage.to_wire(data)
        back = RLXMessage.from_wire(wire)
        assert len(data) == len(back)
        assert data['arg1'] == back['arg1']
        assert data['array'] == back['array']
        npt.assert_allclose(data['nparray'], back['nparray'])
