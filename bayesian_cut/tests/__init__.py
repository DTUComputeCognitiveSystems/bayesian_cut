"""
The :mod:`bayesian_cut.tests` module contains an installation test script, which contains a range of tests created for
the pytest package that ensure the correct installation of the bayesian_cut package and subsequently the recreation of
predefined results for all methods. This is especially designed to verify the validity of results after making changes
to the source code of the implemented algorithms.
"""

#!/usr/bin/env python3
#
# -*- coding: utf-8 -*-
#
# Author: Laurent Vermue <lauve@dtu.dk>
#
#
# License: 3-clause BSD

from . import test_bayesian_cut

__all__ = ["test_bayesian_cut"]