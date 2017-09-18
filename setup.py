import codecs
import os
import re
from setuptools import setup
from setuptools import find_packages
from subprocess import Popen, PIPE
import platform
from ctypes.util import find_library


def read(*path):
    here = os.path.dirname(__file__)
    with codecs.open(os.path.join(here, *path), encoding='utf-8') as fp:
        return fp.read()


def find_nvcc():
    proc = Popen(['which', 'nvcc'], stdout=PIPE, stderr=PIPE)
    out, err = proc.communicate()
    out = out.decode().strip()
    if len(out) > 0:
        return os.path.dirname(out)
    else:
        return None


def find_cuda():
    cuda_home = os.getenv('CUDA_HOME', '/usr/local/cuda')
    if not os.path.exists(cuda_home):
        # We use nvcc path on Linux and cudart path on macOS
        osname = platform.system()
        if osname == 'Linux':
            cuda_path = find_nvcc()
        else:
            cudart_path = find_library('cudart')
            if cudart_path is not None:
                cuda_path = os.path.dirname(cudart_path)
            else:
                cuda_path = None
        if cuda_path is not None:
            cuda_home = os.path.dirname(cuda_path)
        else:
            cuda_home = None
    return cuda_home is not None


# Build-specific dependencies.
extras = {
    'keras': ['keras==1.2.1'],
    'wsproxy': ['autobahn==0.17.2', 'Twisted==17.1.0'],
    'testing': ['pytest', 'pytest-cov', 'pytest-xdist', 'flake8', 'mock']
}

# Meta dependency groups.
all_deps = []
for group_name in extras:
    all_deps += extras[group_name]
extras['all'] = all_deps

VERSION = re.search(
    r'^__version__ = [\'"]([^\'"]*)[\'"]',
    read('relaax/__init__.py')
).group(1)

install_requires = [
    'ruamel.yaml',
    'grpcio_tools',
    'grpcio',
    'boto3',
    'pillow',
    'numpy>=1.11.0',
    'scipy',
    'psutil',
    'honcho==0.7.1',
    'h5py',
    'click',
    'future'
]

# Determine TensorFlow version to be installed
try:
    import tensorflow
    # TensorFlow installed, nothing to do
except ImportError:
    # TensorFlow not found, try find CUDA

    is_cuda = find_cuda()

    if is_cuda:
        print("CUDA found, installing TensorFlow with GPU support")
        tensorflow_package = 'tensorflow-gpu'
    else:
        # Can't find CUDA or NVCC install, install CPU version
        print("CUDA not found, using CPU-only version of TensorFlow")
        tensorflow_package = 'tensorflow'

    install_requires += ["{}>=1,<2".format(tensorflow_package)]

setup(
    name='relaax',
    version=VERSION,
    description=('Reinforcement Learning framework to facilitate development '
                 'and use of scalable RL algorithms and applications'),
    author='Deep Learn',
    author_email='relaax@dplrn.com',
    url='https://github.com/deeplearninc/relaax',
    license='MIT',
    install_requires=install_requires,
    extras_require=extras,
    entry_points={
        'console_scripts': [
            'relaax=relaax.cmdl.cmdl:cmdl',
            'relaax-metrics-server=relaax.server.metrics_server.metrics_server:main',
            'relaax-parameter-server=relaax.server.parameter_server.parameter_server:main',
            'relaax-rlx-server=relaax.server.rlx_server.rlx_server:main',
            'relaax-wsproxy=relaax.server.wsproxy.wsproxy:main'
        ]
    },
    packages=find_packages()
)
