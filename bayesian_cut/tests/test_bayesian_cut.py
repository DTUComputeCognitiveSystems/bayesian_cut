"""
Script to test if all algorithms perform as intended using pytest
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author:   Laurent Vermue <lauve@dtu.dk>
#
# License: 3-clause BSD

# define Parameters
rand_seed = 25

# initialize
import numpy as np
import os
print(os.getcwd())
from . import load_data

if __name__ == '__main__':
    test_dir = load_data.data_path()
else:
    test_dir = os.path.dirname(__file__)

# First check the availability of test files
assert load_data.check_test_files() == 0, "Not all required files could be obtained. Tests cannot be performed"

import pickle

X, Y, unique_samples, unique_samples_eval_BC, unique_samples_eval_dcSBM, unique_samples_eval_ModCut,\
               unique_samples_eval_RatioCut, unique_samples_eval_NormCut, sim_matrix = pickle.load(open('test_data.pkl',
                                                                                                        'rb'))

#%% Testing the bayesian cut model class
def test_BC_single_thread():
    from bayesian_cut.cuts.bayesian_models import Model
    n_samples_per_chain = 25
    n_chains = 2
    C = 2

    model_params = {
        'alpha_in': 1e-2,
        'beta_in': 1e-2,
        'alpha_out': 1e-2,
        'beta_out': 1e-2,
        'b': 1,
        'gamma': 3
    }

    BC = Model('ShiftedApproximateBayesianCommunityDetection',
               X,
               model_params,
               Y=Y,
               C=C,
               block_sampling=False,
               marginalize_phi=True
               )

    BC.add_chains(number_of_chains=n_chains)

    BC.run_chains(n_samples=n_samples_per_chain,
                  n_prior_updates=20,
                  verbose=False,
                  save_all_samples=False,
                  parallel=False
                  )


def test_BC_multi_thread():
    from bayesian_cut.cuts.bayesian_models import Model
    n_samples_per_chain = 25
    n_chains = 2
    C = 2

    model_params = {
        'alpha_in': 1e-2,
        'beta_in': 1e-2,
        'alpha_out': 1e-2,
        'beta_out': 1e-2,
        'b': 1,
        'gamma': 3
    }

    BC = Model('ShiftedApproximateBayesianCommunityDetection',
               X,
               model_params,
               Y=Y,
               C=C,
               block_sampling=False,
               marginalize_phi=True
               )

    BC.add_chains(number_of_chains=n_chains)

    BC.run_chains(n_samples=n_samples_per_chain,
                  n_prior_updates=20,
                  verbose=False,
                  save_all_samples=False,
                  parallel=True
                  )

def test_dcSBM_multi_thread():
    from bayesian_cut.cuts.bayesian_models import Model
    n_samples_per_chain = 25
    n_chains = 2
    C = 2

    model_params = {
        'alpha_in': 1e-2,
        'beta_in': 1e-2,
        'alpha_out': 1e-2,
        'beta_out': 1e-2,
        'b': 1,
        'gamma': 3
    }

    dcSBM = Model('BayesianStochasticBlockmodelSharedEtaOut',
               X,
               model_params,
               Y=Y,
               C=C,
               block_sampling=False,
               marginalize_phi=True
               )

    dcSBM.add_chains(number_of_chains=n_chains)

    dcSBM.run_chains(n_samples=n_samples_per_chain,
                  n_prior_updates=20,
                  verbose=False,
                  save_all_samples=False,
                  parallel=True
                  )

#%% Testing the spectral methods
def test_RatioCut():
    from bayesian_cut.cuts.spectral_models import RatioCut
    from sklearn.metrics import adjusted_rand_score as ARI
    rc = RatioCut(X)
    rc.run()
    assert(ARI(Y, rc.z_) == 1)

def test_NormCut():
    from bayesian_cut.cuts.spectral_models import NormCutSM
    from sklearn.metrics import adjusted_rand_score as ARI
    nc = NormCutSM(X)
    nc.run()
    assert(ARI(Y, nc.z_) == 1)

def test_ModCut():
    from bayesian_cut.cuts.spectral_models import NewmanModularityCut
    from sklearn.metrics import adjusted_rand_score as ARI
    modcut = NewmanModularityCut(X)
    modcut.run()
    assert(ARI(Y, modcut.z_) == 1)

#%% Testing the utility functions
def test_RatioCut_Costfunction():
    from bayesian_cut.utils.utils import calc_ratiocut_scores
    scores = calc_ratiocut_scores(unique_samples, X)
    assert(np.array_equal(scores, unique_samples_eval_RatioCut))

def test_NormCut_Costfunction():
    from bayesian_cut.utils.utils import calc_normcut_scores
    scores = calc_normcut_scores(unique_samples, X)
    assert(np.array_equal(scores, unique_samples_eval_NormCut))

def test_ModCut_Utilityfunction():
    from bayesian_cut.utils.utils import calc_modularity_scores
    scores = calc_modularity_scores(unique_samples, X)
    assert(np.array_equal(scores, unique_samples_eval_ModCut))

def test_BC_likelihood():
    from bayesian_cut.cuts.bayesian_models import Model
    from bayesian_cut.utils.utils import recalculate_prob_for_z
    n_samples_per_chain = 25
    n_chains = 2
    C = 2

    model_params = {
        'alpha_in': 1e-2,
        'beta_in': 1e-2,
        'alpha_out': 1e-2,
        'beta_out': 1e-2,
        'b': 1,
        'gamma': 3
    }

    BC = Model('ShiftedApproximateBayesianCommunityDetection',
               X,
               model_params,
               Y=Y,
               C=C,
               block_sampling=False,
               marginalize_phi=True
               )

    BC.add_chains(number_of_chains=n_chains)

    recalc_log_lik = recalculate_prob_for_z(BC, unique_samples)
    assert(np.array_equal(recalc_log_lik, unique_samples_eval_BC))


def test_dcSBM_likelihood():
    from bayesian_cut.cuts.bayesian_models import Model
    from bayesian_cut.utils.utils import recalculate_prob_for_z
    n_samples_per_chain = 25
    n_chains = 2
    C = 2

    model_params = {
        'alpha_in': 1e-2,
        'beta_in': 1e-2,
        'alpha_out': 1e-2,
        'beta_out': 1e-2,
        'b': 1,
        'gamma': 3
    }

    dcSBM = Model('BayesianStochasticBlockmodelSharedEtaOut',
                  X,
                  model_params,
                  Y=Y,
                  C=C,
                  block_sampling=False,
                  marginalize_phi=True
                  )

    dcSBM.add_chains(number_of_chains=n_chains)

    recalc_log_lik = recalculate_prob_for_z(dcSBM, unique_samples)
    assert(np.array_equal(recalc_log_lik, unique_samples_eval_dcSBM))
