#!/usr/bin/env python

from setuptools import setup
from setuptools import find_packages

__minimum_numpy_version__ = '1.19.0'

setup_requires = ['numpy<' + __minimum_numpy_version__,
                  'h5parm']

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='bayes_gain_screens',
      version='0.0.1',
      description='Bayesian directional TEC modelling for LOFAR HBA',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/joshuaalbert/bayes_gain_screens",
      author='Joshua G. Albert',
      author_email='albert@strw.leidenuniv.nl',
      setup_requires=setup_requires,
      tests_require=[
          'pytest>=2.8',
      ],
      package_dir={'': './'},
      packages=find_packages('./'),
      package_data={'bayes_gain_screens': ['arrays/*', 'flagging_models/*', 'steps/*']},
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: Apache Software License",
          "Operating System :: OS Independent",
      ],
      python_requires='>=3.6',
      )
