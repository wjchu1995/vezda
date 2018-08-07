#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from setuptools import setup, find_packages
import textwrap

if sys.version_info <= (2,7):
    sys.exit(textwrap.dedent(
            '''
            Sorry, Python <= 2.7 is not supported. Vezda requires Python >= 3.
            (Try obtaining the Anaconda Python distribution at:
                
                https://www.anaconda.com/download/
            
            for example.)
            '''))

with open('README.md', 'r') as fh:
     long_description = fh.read()

setup(name = 'vezda',
      version = '0.4.3',
      description = 'A set of command-line tools for imaging with the linear sampling method',
      long_description = long_description,
      long_description_content_type = 'text/markdown',
      python_requires = '>=3',
      classifiers = [
              'Development Status :: 4 - Beta',
              'Intended Audience :: Science/Research',
              'License :: OSI Approved :: Apache Software License',
              'Programming Language :: Python :: 3',
              'Programming Language :: Python :: 3.4',
              'Programming Language :: Python :: 3.5',
              'Programming Language :: Python :: 3.6',
              'Topic :: Scientific/Engineering :: Mathematics',
              'Topic :: Scientific/Engineering :: Physics'
              ],
      keywords = 'imaging shape reconstruction boundary detection linear sampling method',
      url = 'https://github.com/aaronprunty/vezda',
      author = 'Aaron C. Prunty',
      license = 'Apache License, Version 2.0',
      packages = find_packages(),
      package_data = {'vezda': ['examples/*', 'docs/*']},
      include_package_data = True,
      install_requires = ['argparse',
                          'matplotlib',
                          'numpy',
                          'pathlib',
                          'scipy',
                          'scikit-image',
                          'tqdm'],
      entry_points = {
              'console_scripts': [
                      'vzdata = vezda.setDataPath:cli',
                      'vzgrid = vezda.setSamplingGrid:cli',
                      'vzhome = vezda.home:cli',
                      'vzimage = vezda.plotImage:cli',
                      'vzpicard = vezda.Picard:cli',
                      'vzsolve = vezda.Solve:cli',
                      'vzsvd = vezda.SVD:cli',
                      'vzwiggles = vezda.plotWiggles:cli',
                      'vzwindow = vezda.setWindow:cli'
                      ]
              },
      zip_safe = False)
