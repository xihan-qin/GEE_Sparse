# -*- coding: utf-8 -*-

from setuptools import setup
from setuptools import find_packages

# Package meta-data.
NAME = 'GEEsparse'
DESCRIPTION = 'Graph Encoder Embedding for large sparse graphs.'
URL = 'https://github.com/xihan-qin/GEE_Sparse'
EMAIL = 'shenc@udel.edu, xihan@udel.edu'
AUTHOR = 'Cencheng Shen, Xihan Qin'
REQUIRES_PYTHON = '>=3.8.5'
VERSION = '0.1.0'

# What packages are required for this module to be executed?
REQUIRED = [
    # 'requests', 'maya', 'records',
]
    
setup(name='GEEsparse',
      version= VERSION,
      description= DESCRIPTION,
      author= AUTHOR,
      author_email= EMAIL,
      python_requires=REQUIRES_PYTHON,
      url= URL,
      install_requires=['numpy',
                        'tensorflow',
                        'scipy',
                        'sklearn'
                        ],
      package_data={'pygcn': ['README.md']},
      packages=find_packages())