import os
import re
import codecs
from setuptools import setup
from setuptools import find_packages


def read(*path):
    here = os.path.dirname(__file__)
    with codecs.open(os.path.join(here, *path), encoding='utf-8') as fp:
        return fp.read()


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
        'autobahn==0.17.2',
        'Twisted==17.1.0',
        'ruamel.yaml',
        'grpcio_tools',
        'grpcio',
        'boto3',
        'pillow',
        'numpy',
        'scipy',
        'psutil',
        'honcho==0.7.1',
        'keras==1.2.1',
        'h5py',
        'tensorflow'
    ],
    extras_require={
        'testing': [
            'pytest',
            'pytest-cov',
            'pytest-xdist',
            'flake8',
            'mock']
    },
    entry_points={
        'console_scripts': [
            'relaax=relaax.common.python.cmdl.cmdl_run:main',
            'relaax-parameter-server=relaax.server.parameter_server.parameter_server:main',
            'relaax-rlx-server=relaax.server.rlx_server.rlx_server:main'
        ]
    },
    packages=find_packages())
