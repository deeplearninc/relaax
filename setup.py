#!/usr/bin/env python

from distutils.core import setup

setup(
    name='relaax',
    version='0.2',
    description='RELAAX solution',
    author='Stanislav Volodarskiy',
    author_email='stanislav@dplrn.com',
    url='http://www.dplrn.com/',
    packages=['relaax'],
    install_requires=[
        'autobahn==0.17.2',
        'Twisted==17.1.0',
        'ruamel.yaml',
        'futures',
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
        'mock',
        'tensorflow'
    ],
    provides=['relaax'],
    scripts=[
        'bin/relaax-parameter-server',
        'bin/relaax-rlx-server',
        'bin/relaax'
    ]
)
