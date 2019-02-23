"""
The Bayesian Cut for Python
===========================

The Bayesian Cut Python package provides an easy to use API for the straight-forward application of Bayesian network
cuts using a full Bayesian inference framework based on the Gibbs-Sampler using the degree corrected Stochastic
Blockmodel (dc-SBM) or the Bayesian Cut (BC).
Furthermore it provides modularity, ratio-cut and norm cut based spectral network cut methods.
It also provides a rich visualization library that allow an easy analysis of posterior solution landscapes and network
cuts obtained by the various methods.

The aim of the package is to provide an easily accessible user interface to this method.
"""

#!/usr/bin/env python3
#
# -*- coding: utf-8 -*-
#
# author:   Laurent Vermue <lauve@dtu.dk>
#           Maciej Korzepa <mjko@dtu.dk>
#           Petr Taborsky <ptab@dtu.dk>
#           Morten Mørup <mmor@dtu.dk>
#
# License: 3-clause BSD

from . import utils, data, cuts

__all__ = ["cuts", "data", "utils"]

__version__ = "0.1.0-beta"
