"""
Script to check the availability of test files
"""

#!/usr/bin/env python3
#
# -*- coding: utf-8 -*-
#
# Author: Laurent Vermue <lauve@dtu.dk>
#
#
# License: 3-clause BSD

import os
from ..data.load import file_checker

TEST_FILES = ['test_data.pkl']

GITHUB_TESTDIR = 'https://github.com/DTUComputeCognitiveSystems/bayesian_cut/raw/master/bayesian_cut/tests'

def data_path():
    path = os.path.dirname(os.path.abspath(__file__))
    return path

def check_test_files():
    file_path = data_path()
    for file in TEST_FILES:
        abs_file_path = os.path.join(file_path, file)
        if file_checker(file, abs_file_path, GITHUB_TESTDIR) == 0:
            pass
        else:
            print('File {0} was not available and could not be downloaded'.format(file))
            return 1
    return 0
