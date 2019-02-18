#!/usr/bin/env python3
#
# -*- coding: utf-8 -*-
#
# Author:   Laurent Vermue <lauve@dtu.dk>
#           Maciej Korzepa <mjko@dtu.dk>
#           Petr Taborsky <ptab@dtu.dk>
#
#
# License: 3-clause BSD

from setuptools import setup, find_packages

import bayesian_cut

NAME = "bayesian_cut"
VERSION = bayesian_cut.__version__
DESCRIPTION = "An implementation of bayesian cut methods"
URL = 'https://github.com/DTUComputeCognitiveSystems/bayesian_cut'
AUTHORS = "Laurent Vermue, Maciej Korzepa, Petr Taborsky, Morten Mørup"
AUTHOR_MAILS = "<lauve@dtu.dk>, <mjko@dtu.dk>, <ptab@dtu.dk>, <mmor@dtu.dk>"
LICENSE = 'new BSD'

# This is the lowest tested version. Below might work as well
NUMPY_MIN_VERSION = '1.13.3'
SCIPY_MIN_VERSION = '1.0.0'
SCIKIT_LEARN_MIN_VERSION = '0.20.0'
NETWORKX_MIN_VERSION = '2.0'
JOBLIB_MIN_VERSION = '0.12.3'

def setup_package():
    with open('README.rst') as f:
        LONG_DESCRIPTION = f.read()
        LONG_DESCRIPTION_CONTENT_TYPE = 'text/x-rst'

    setup(name=NAME,
          version=VERSION,
          description=DESCRIPTION,
          long_description=LONG_DESCRIPTION,
          long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
          url=URL,
          author=AUTHORS,
          author_email=AUTHOR_MAILS,
          packages = find_packages(),
          license=LICENSE,
          classifiers=['Intended Audience :: Science/Research',
                       'Intended Audience :: Developers',
                       'License :: OSI Approved',
                       'Programming Language :: Python',
                       'Topic :: Software Development',
                       'Topic :: Scientific/Engineering',
                       'Operating System :: Microsoft :: Windows',
                       'Operating System :: POSIX',
                       'Operating System :: Unix',
                       'Operating System :: MacOS',
                       'Programming Language :: Python :: 3.5',
                       'Programming Language :: Python :: 3.6',
                       'Programming Language :: Python :: 3.7',
                       'Development Status :: 4 - Beta',
                       # 'Development Status :: 5 - Production/Stable'
                       ],
          install_requires=[
              'numpy>={0}'.format(NUMPY_MIN_VERSION),
              'scipy>={0}'.format(SCIPY_MIN_VERSION),
              'scikit-learn>={0}'.format(SCIKIT_LEARN_MIN_VERSION),
              'networkx>={0}'.format(NETWORKX_MIN_VERSION),
              'joblib>={0}'.format(JOBLIB_MIN_VERSION)
                ],
          extras_require={
              'tests': [
                  'pytest'],
              'docs': [
                  'sphinx >= 1.6',
                  'sphinx_rtd_theme',
                  'nbsphinx',
                  'nbsphinx_link'
                    ],
              'extras': [
                  'matplotlib',
                  'plotly',
                  'naturalneighbor',
                  'seaborn'
              ],
          },
          python_requires='>=3.5',
          )

if __name__ == '__main__':
    setup_package()
