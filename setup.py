#!/usr/bin/env python

from setuptools import setup
from setuptools import find_packages


requirements = [
    'numpy',
    'h5parm',
    'jaxns>=1.0.0',
    'scipy',
    'numpy',
    'matplotlib',
    'cmocean',
    'astropy',
    'pyregion',
    'dm-sonnet',
    'tensorflow',
    'graph_nets',
    'tqdm',
    'sympy',
    'pyregion',
    'pyparsing',
    'jax',
    'jaxlib',
    'tables'
]

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='bayes_gain_screens',
    version='0.0.1',
    description='Bayesian directional TEC modelling for LOFAR HBA',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/joshuaalbert/bayes_gain_screens",
    author='Joshua G. Albert',
    author_email='albert@strw.leidenuniv.nl',
    install_requires=requirements,
    tests_require=[
        'pytest>=2.8',
    ],
    package_dir={'': './'},
    packages=find_packages('./'),
    package_data={
        'bayes_gain_screens': [
            'arrays/*',
            'flagging_models/*',
            'steps/*',
            'steps/templates/*'
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
