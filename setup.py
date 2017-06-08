import codecs
import os
import re
from setuptools import setup
from setuptools import find_packages


def read(*path):
    here = os.path.dirname(__file__)
    with codecs.open(os.path.join(here, *path), encoding='utf-8') as fp:
        return fp.read()

# Build-specific dependencies.
extras = {
    'keras': ['keras==1.2.1'],
    'wsproxy': ['autobahn==0.17.2','Twisted==17.1.0'],
    'testing': ['pytest','pytest-cov','pytest-xdist','flake8','mock']
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

setup(
    name='relaax',
    version=VERSION,
    description=('Reinforcement Learning framework to facilitate development '
                 'and use of scalable RL algorithms and applications'),
    author='Deep Learn',
    author_email='relaax@dplrn.com',
    url='https://github.com/deeplearninc/relaax',
    license='MIT',
    install_requires=[
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
        'tensorflow==1.1.0',
        'click',
        'future'
    ],
    extras_require=extras,
    entry_points={
        'console_scripts': [
            'relaax=relaax.cmdl.cmdl:cmdl',
            'relaax-parameter-server=relaax.server.parameter_server.parameter_server:main',
            'relaax-rlx-server=relaax.server.rlx_server.rlx_server:main',
            'relaax-wsproxy=relaax.server.wsproxy.wsproxy:main'
        ]
    },
    packages=find_packages()
)
