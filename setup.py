import os, glob

from setuptools import setup, find_packages

HERE = os.path.dirname(os.path.abspath(__file__))


# with open('requirements.txt') as f:
#     requirements = f.read().splitlines()
#     requirements = [r for r in requirements if not r.startswith('-e')]


# packages = ['djow'] + glob.glob('djow_*')

setup(
    name='HarmonicPatterns',
    version='0.0.1',
    # install_requires=requirements,
    # packages=packages,
    url='https://github.com/djoffrey/HarmonicPatterns',
    author='djoffrey',
    author_email='joffrey.oh@gmail.com',
    description='A Python Library for Harmonic Trading'
)
