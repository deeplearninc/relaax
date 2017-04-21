import codecs
import os
import re
from setuptools import setup
from setuptools import find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install


def build_bridge():
    # this lazy import is valid only on post-install step
    import grpc.tools.protoc

    bridge_dir = 'relaax/server/common/bridge'
    exit_code = grpc.grpc_tools.protoc.main([
        '-I%s' % bridge_dir,
        '--python_out=.',
        '--grpc_python_out=.',
        os.path.join(bridge_dir, 'bridge.proto')
    ])
    if exit_code != 0:
        raise Exception('cannot compile a GRPC bridge')


class PostDevelopCommand(develop):
    def run(self):
        compile_bridge()
        develop.run(self)


class PostInstallCommand(install):
    def run(self):
        compile_bridge()
        install.run(self)


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
        'tensorflow==1.0.1'
    ],
    extras_require={
        'keras': [
            'keras==1.2.1'
        ],
        'wsproxy': [
            'autobahn==0.17.2',
            'Twisted==17.1.0'
        ],
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
    packages=find_packages(),
    cmdclass={
        'develop': PostDevelopCommand,
        'install': PostInstallCommand
    }
)
