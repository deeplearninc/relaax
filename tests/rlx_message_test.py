from builtins import object
import numpy as np
import numpy.testing as npt

from relaax.common.rlx_message import RLXMessage
from relaax.common.rlx_message import RLXMessageImage
from PIL import Image

class TestRLXMessage(object):

    def test_to_wire_and_back(self):
        npar1 = np.array([1.001, 2.002, 3.003], dtype=np.float32)
        npar2 = np.array([[1.001, 2.002, 3.003], [1.001, 2.002, 3.003]])
        npar2empty = np.array([[], []])
        npar1empty = np.array([])

        imageJPG = Image.open("./tests/fixtures/testimage.jpg")
        imageBMP = Image.open("./tests/fixtures/testimage.bmp")

        data = {'arg1': 1, 'arg_1': -1, "args": "test1", "argn": -234, "argbt": True, "argbf": False, "argnone": None,
                "array_empty": [], 'array': [4.02, 5.2, 6.006],
                'nparray1': npar1, 'nparray2': npar2, 'nparray1empty': npar1empty, 'nparray2empty': npar2empty,
                'image_jpg': RLXMessageImage(imageJPG), 'image_bmp': RLXMessageImage(imageBMP)}
        wire = RLXMessage.to_wire(data)
        back = RLXMessage.from_wire(wire)
        assert len(data) == len(back)
        for key in data:
            if isinstance(data[key], np.ndarray):
                assert type(data[key]) == type(back[key])
                npt.assert_allclose(data[key], back[key])
            elif isinstance(data[key], RLXMessageImage):
                assert isinstance(back[key], np.ndarray)
                ndimage = np.asarray(data[key].image)
                npt.assert_allclose(ndimage, back[key])
            else:
                assert type(data[key]) == type(back[key])
                assert data[key] == back[key]
