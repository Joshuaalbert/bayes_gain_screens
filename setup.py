#!/usr/bin/env python

from setuptools import setup
from setuptools import find_packages

__minimum_numpy_version__ = '1.17.0'
__minimum_tensorflow_version__ = '1.15.0'

setup_requires = ['numpy>=' + __minimum_numpy_version__, 
'tensorflow>='+__minimum_tensorflow_version__]

setup(name='bayes_gain_screens',
      version='0.0.1',
      description='Bayesian directional TEC model',
      author=['Josh Albert'],
      author_email=['albert@strw.leidenuniv.nl'],
    setup_requires=setup_requires,  
    tests_require=[
        'pytest>=2.8',
    ],
    package_data= {'bayes_gain_screens':['arrays/*', 'flagging_models/*']},
   package_dir = {'':'./'},
   packages=find_packages('./')
     )

