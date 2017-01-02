#!/usr/bin/env python

from distutils.core import setup

setup(
    name='relaax',
    version='1.0',
    description='RELAAX solution',
    author='Stanislav Volodarskiy',
    author_email='stanislav@dplrn.com',
    url='http://www.dplrn.com/',
    packages=['relaax'],
    requires=[
        'ruamel.yaml',
        'futures',
        'grpcio-tools',
        'grpcio',
        'boto3',
        'pillow',
        'scipy'
    ],
    provides=['relaax'],
    scripts=[
        'bin/relaax-parameter-server',
        'bin/relaax-rlx-server'
    ]
)
