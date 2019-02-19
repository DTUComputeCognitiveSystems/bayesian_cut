#!/usr/bin/env python3
#
# -*- coding: utf-8 -*-
#
# Author:   Maciej Korzepa <mjko@dtu.dk>
#           Laurent Vermue <lauve@dtu.dk>
#           Petr Taborsky <ptab@dtu.dk>
#
# License: 3-clause BSD

import collections
import warnings
import numpy as np
import time
from functools import lru_cache
from joblib import Parallel, delayed
from scipy.integrate import quad
from scipy.special import loggamma, gamma, gammaincc
from scipy.stats import beta as beta_dist, gamma as gamma_dist
from sklearn import metrics

__all__ = ['Model']


class _CommunityDetectionChain(object):
    """
    Base Class Implementing the Gibbs Sampler
    -----------------------------------------

    Parameters
    ----------
    X : sparse scipy matrix, shape(number of nodes, number of nodes)
        The adjacency matrix

    model_params : dict
        Dictionary containing values for the parameter keys alpha_in, alpha_out, beta_in, beta_out, b and gamma.
        If a value for a key is None, the parameter is inferred.

    C : int, optional, default: 2
        The number of clusters to infer (current implementation only supports 2 clusters)

    Y : numpy array, optional
        Ground truth vector for the group assignments. This allows the sampler to report the Adjusted Random Index of
        new samples.

    shared_b : boolean, optional, default: False

    block_sampling : boolean, optional, default: False
        All leaf nodes will be aggregated to a block and will not be considered for sampling

    marginalize_phi : boolean, optional, default: True
        Defines if the phi term of the likelihood function should be marginalized during calculation

    chain_id : int, optional, default: 1
        Chain identifier of the object

    Attributes
    ----------
    samples_ : numpy array, shape(number of samples + number of burn-in-samples, number of nodes)
        All samples from sampling

    samples_log_lik_ : numpy array, shape(number of samples + number of burn-in-samples, 1)
        Log-likelihood of each sample

    burn_in_samples_ : numpy array, shape(number of burn-in-samples, number of nodes)
        All samples from the burn-in-phase

    burn_in_samples_log_lik_ : numpy array, shape(number of burn-in-samples, number of nodes)
        Log-likelihood for each burn-in-sample

    post_samples_ : numpy array, shape(number of samples, number of nodes)
        All samples after burn-in-phase

    post_samples_log_lik_ : numpy array, shape(number of samples, 1)
        Log-likelihood for each sample after burn-in-phase

    anneal_max_log_lik_z_ : numpy array, shape(number of nodes,)
        Group assignment vector with highest log-likelihood obtained by deterministic optimization

    anneal_max_log_lik_ : float
        Log-likelihood for maximum log-likelihood group vector obtained by deterministic optimization

    anneal_max_log_lik_params_ : dict
        Dictionary containing the parameters for the group assignment vector with highest log-likelihood obtained by
        deterministic optimization

    sampled_params_ : dict
        Dictionary containing numpy arrays with shape(number of samples + number of burn-in-samples, 1) for each
        parameter inferred

    burn_in_sampled_params_ : dict
        Dictionary containing numpy arrays with shape(number of burn-in-samples, 1) for each parameter inferred

    post_sampled_params_ : dict
        Dictionary containing numpy arrays with shape(number of samples, 1) for each parameter inferred

    max_log_lik_z_ : numpy array, shape(number of nodes,)
        Group assignment vector with highest log-likelihood obtained by Gibbs sampler

    max_log_lik_ : float
        Log-likelihood for maximum log-likelihood group vector obtained by Gibbs sampler

    max_log_lik_params_ : dict
        Dictionary containing the parameters for the group assignment vector with highest log-likelihood obtained by
        Gibbs sampler

    z_all_ : numpy array, shape(number of total visited solutions, number of nodes)
        (only if the chain was run with save_all_samples=True)
        Matrix containing all samples for all solutions visited during sampling of this chain.

    px_z_all_ : numpy array, shape(number of total visited solutions, 1)
        (only if the chain was run with save_all_samples=True)
        Log-likelihood for each solution visited.
    """

    def __init__(self, X, model_params, C=2, Y=None, shared_b=False, block_sampling=False,
                 marginalize_phi=True, chain_id=1):
        self.C = C
        self.marginalize_phi = marginalize_phi
        self.shared_b = shared_b
        self.sort_communities = True

        self.saved_integrals_ = {}
        self.X = X.copy()
        self._X_blocks = self.X.copy()
        self._k = np.squeeze(np.asarray(X.sum(axis=1)))
        self._n = X.shape[0]
        self._N = int(np.sum(self._k) / 2)
        self._self_links = self.X.diagonal() / 2  # keep true self-link count in a separate array
        self.X.setdiag(0)  # don't store self link in the main matrix

        # faster way to access data from adjacency matrix
        self._X_indices = {}
        self._X_data = {}
        for i in range(X.shape[0]):
            row = self.X[i]
            self._X_indices[i] = row.indices
            self._X_data[i] = row.data

        if block_sampling:
            self._sample_indices = np.where((self.X > 0).sum(1) > 1)[0]
        else:
            self._sample_indices = list(range(self._n))
        self._X_blocks[:, self._sample_indices] = 0
        self._X_blocks.eliminate_zeros()
        self._X_blocks_sums = self._X_blocks.copy()
        self._X_blocks_sums.data = self._self_links[self._X_blocks_sums.indices]
        self._X_blocks_self_links_sum = self._X_blocks_sums.sum(1).A1
        self._X_blocks_sums = self._X_blocks_self_links_sum + self._X_blocks.sum(1).A1 + self._self_links
        self._X_blocks.setdiag(1)
        X_blocks_dict = {}
        for i in self._sample_indices:
            X_blocks_dict[i] = self._X_blocks[i].indices
        self._X_blocks = X_blocks_dict

        self.Y = Y if Y is not None else np.zeros((self._n,), dtype=np.int)

        self._node_aux = None
        self._links_nodes = None
        self.samples_ = None
        self.samples_log_lik_ = None
        self.n_samples = None
        self.n_prior_updates = None
        self.save_all_samples = None

        self.n_burn_in_samples = None
        self.burn_in_samples_ = None
        self.burn_in_samples_log_lik_ = None
        self.post_samples_ = None
        self.post_samples_log_lik_ = None

        self._z = None
        self._max_log_list = []
        self._Nc = None
        self._N_out = None
        self._nc = None
        self._Kc = None

        if not self.marginalize_phi:
            self._k_log_k = np.sum(self._k * np.log(self._k + 1))  # will crash if there are nodes with no links

        self.chain_id = chain_id
        self._log_C_phi = 0
        self._old_max_jll = -np.inf
        self.anneal_max_log_lik_z_ = None
        self.anneal_max_log_lik_ = None
        self.anneal_max_log_lik_params_ = None
        self.max_log_lik_z_ = None
        self.max_log_lik_ = None
        self.max_log_lik_params_ = None

        # Allow saving of all samples when required
        self.px_z_all_ = []
        self.z_all_ = []

        # Global verbose level
        self.verbose = False

        # Tracking of numerical errors
        self._numint_error = None

        self.model_params = {}
        self._model_params_b = None if self.shared_b else np.array([np.nan] * self.C)  # collects bs into a list
        self._infer_params = set()
        self.sampled_params_ = {}
        self.burn_in_sampled_params_ = {}
        self.post_sampled_params_ = {}
        self._log_joint_prior = 0

        if model_params['gamma'] is None:
            model_params['gamma'] = 1.0
            self._infer_params.add('gamma')
        if model_params['b'] is None:
            model_params['b'] = 0.999
            if self.shared_b:
                self._infer_params.add('b')
            else:
                for i in range(1, C + 1):
                    self._infer_params.add('b_{}'.format(i))

        for pname, value in model_params.items():
            if pname != 'b' or self.shared_b:
                self.set_param(pname, value, False)
            else:
                for i in range(1, C + 1):
                    self.set_param('b_{}'.format(i), value, False)
        self.calc_joint_prior()

    def log_lik(self):
        if (self._nc > 0).sum() < self.C:
            return -np.inf

        return np.sum(self.log_lik_terms())

    def log_lik_terms(self):
        eta_term = self.evaluate_eta_term()
        phi_term = np.sum(self._Kc * np.log(self._nc))

        if self.marginalize_phi:
            phi_term += np.sum(loggamma(self._nc * self.model_params['gamma'])) \
                        - np.sum(loggamma(self._Kc + self._nc * self.model_params['gamma'])) \
                        + self._log_C_phi
        else:
            phi_term += -np.sum(self._Kc * np.log(self._Kc)) + self._k_log_k

        return eta_term, phi_term

    def evaluate_eta_term(self):
        pass

    def calc_log_lik_for_z(self, z, joint_with_prior=False):
        self.set_z(z)
        return self.log_lik() + (self._log_joint_prior if joint_with_prior else 0)

    def calculate_unconstrained_ml_etas(self):
        nc_pow2 = self._nc ** 2
        nc_pow2_sum = np.sum(nc_pow2)
        n_out_pow2 = self._n ** 2 - nc_pow2_sum
        eta_in = 2 * self._Nc / nc_pow2
        eta_out = 2 * self._N_out / n_out_pow2

        return eta_in, eta_out

    def set_z(self, z, params=None):
        self._z = z.copy()
        self._calc_node_assignments()
        self.calc_links_nodes()
        if params is not None:
            for pname, value in params.items():
                self.set_param(pname, value)
            self.calc_joint_prior()

    def ensure_correct_z(self, z, raise_if_incorrect=False):
        for i in self._sample_indices:
            if raise_if_incorrect and not np.all(z[self._X_blocks[i]] == z[i]):
                raise Exception('Incorrect starting point.')
            else:
                z[self._X_blocks[i]] = z[i]

        return z

    def update_links_nodes_block(self, i, from_group, to_group):
        from_group_links, to_group_links = self._node_aux[[from_group, to_group], i]
        update_node_aux_indices = self._X_indices[i]
        update_node_aux_data = self._X_data[i]
        block_size = self._X_blocks[i].size
        self._nc[to_group] += block_size
        self._nc[from_group] -= block_size
        self._Kc[to_group] += self._k[i] + self._X_blocks_self_links_sum[i] + self._X_blocks_sums[i]
        self._Kc[from_group] -= self._k[i] + self._X_blocks_self_links_sum[i] + self._X_blocks_sums[i]
        self._Nc[to_group] += to_group_links + self._X_blocks_sums[i]
        self._Nc[from_group] -= from_group_links + self._X_blocks_sums[i]
        self._N_out += from_group_links - to_group_links
        self._node_aux[from_group, update_node_aux_indices] -= update_node_aux_data
        self._node_aux[to_group, update_node_aux_indices] += update_node_aux_data

    def update_links_nodes_multiple(self, i_, from_group_, to_group_):
        if not isinstance(i_, collections.Iterable):
            i_ = [i_]
            from_group_ = [from_group_]
            to_group_ = [to_group_]

        for i, from_group, to_group in zip(i_, from_group_, to_group_):
            self.update_links_nodes_block(i, from_group, to_group)

    def calc_links_nodes(self):
        self._Nc = np.zeros(shape=(self.C,))
        self._nc = np.zeros(shape=(self.C,))
        self._Kc = np.zeros(shape=(self.C,))
        for c in range(self.C):
            idx = np.where(self._z == c)[0]
            self._Kc[c] = self._k[idx].sum()
            self._Nc[c] = self.X[idx, :].tocsc()[:, idx].sum() / 2 + self._self_links[idx].sum()
            self._nc[c] = idx.size

        self._N_out = self._N - self._Nc.sum()

    def _calc_node_assignments(self):
        aux = np.zeros((self._n,), dtype=np.bool)
        aux[self._sample_indices] = True
        self._node_aux = np.zeros(shape=(self.C, self._n))
        for c in range(self.C):
            idx = np.where((self._z == c) & aux)[0]
            self._node_aux[c] = self.X[idx, :].sum(axis=0)

    def _sort_communities(self):
        eta_in, _ = self.calculate_unconstrained_ml_etas()
        order = (-eta_in).argsort()
        map_z = dict(zip(order, np.arange(self.C)))
        if np.any(np.diff(order) < 0):
            self._Nc = self._Nc[order]
            self._nc = self._nc[order]
            self._Kc = self._Kc[order]
            self._node_aux = self._node_aux[order]
            self._z = np.vectorize(map_z.__getitem__)(self._z)

    def inner(self, i, stochastic=True):
        init_assign = self._z[i]
        px_z = np.array([-np.inf] * self.C)
        px_z[init_assign] = self.log_lik() if self._old_max_jll is None else self._old_max_jll
        if self.save_all_samples:
            self.px_z_all_.append(px_z[init_assign])
            self.z_all_.append(self._z.copy())

        for k in range(self.C):
            if k != init_assign:
                self.update_links_nodes_block(i, self._z[i], k)
                self._z[self._X_blocks[i]] = k
                px_z[k] = self.log_lik()
                if self.save_all_samples:
                    # Add samples to memory
                    self.z_all_.append(self._z.copy())
                    # Add corresponding probability to memory
                    self.px_z_all_.append(px_z[k])

        max_ll = px_z.copy()
        if stochastic:
            px_z = np.exp(px_z - np.max(px_z))
            px_z /= np.sum(px_z)
            update = np.random.choice(self.C, p=px_z)
        else:
            update = np.argmax(px_z)

        if self._z[i] != update:
            self.update_links_nodes_block(i, self._z[i], update)
            self._z[self._X_blocks[i]] = update

        self._old_max_jll = max_ll[update]

        return max_ll[update]

    def log_priors(self, param, x):
        param_type = param.split('_')[0]
        if param_type == 'gamma':
            return -np.log(x)
        elif param_type == 'b':
            return -np.log(x)

    def calc_joint_prior(self):
        self._log_joint_prior = np.sum([self.log_priors(pname, self.model_params[pname]) for pname in self._infer_params])

    def set_param(self, param, value, update_joint_prior=True):
        if value != self.model_params.get(param):
            self.model_params[param] = value
            self.recalculate_constants_with_param(param)
            if update_joint_prior:
                self.calc_joint_prior()

    def recalculate_constants_with_param(self, param):
        param_type = param.split('_')[0]
        if param_type == 'gamma' and self.marginalize_phi:
            self._log_C_phi = np.sum(loggamma(self._k + self.model_params['gamma'])) - \
                             self._n * loggamma(self.model_params['gamma'])
        elif param_type == 'b':
            if self.shared_b:
                self._model_params_b = self.model_params['b']
            else:
                self._model_params_b[int(param.split('_')[-1]) - 1] = self.model_params[param]

    def jump(self, param_type, old_value):
        beta_n = 200
        if param_type == 'gamma':
            new_value = np.clip(np.exp(np.log(old_value) + np.random.normal(0, 0.1)), 0, 10 * self._n)
            correction = 0
        elif param_type == 'b':
            new_value = beta_dist.rvs(1 + beta_n * old_value, 1 + beta_n * (1 - old_value))
            correction = self.log_priors('b', new_value) - self.log_priors('b', old_value) \
                         + beta_dist.logpdf(old_value, 1 + beta_n * new_value, 1 + beta_n * (1 - new_value)) \
                         - beta_dist.logpdf(new_value, 1 + beta_n * old_value, 1 + beta_n * (1 - old_value))
        else:
            raise NotImplementedError()

        return new_value, correction

    def sample_prior(self, n_prior_samples=20, stochastic=True):
        if len(self._infer_params) == 0:
            return
        prev_log_lik = None
        for n in range(n_prior_samples):
            for pname in self._infer_params:
                old_value = self.model_params[pname]
                old_log_lik = prev_log_lik if prev_log_lik is not None else self.log_lik()
                new_value, correction = self.jump(pname.split('_')[0], old_value)
                self.set_param(pname, new_value, False)
                new_log_lik = self.log_lik()
                p = np.exp(float(new_log_lik - old_log_lik + correction))
                if (stochastic and np.random.uniform(0, 1) > p) or (not stochastic and new_log_lik <= old_log_lik):
                    self.set_param(pname, old_value, False)
                else:
                    prev_log_lik = new_log_lik

        self.calc_joint_prior()

    def print_progress(self, it, sample_time, max_jll=None):
        if it > self.n_burn_in_samples + self.n_samples:
            print(
                "*** Annealing ***\t"
                "Chain {:>2}: {:>4}/{:d}\t{:.2f}s\t".format(self.chain_id, it, self.n_burn_in_samples,
                                                            sample_time),
                "ARI: {:.3f}".format(metrics.adjusted_rand_score(self.Y, self._z)),
                "\t{:^25}".format('New max lik-log: {:.2f}'.format(max_jll) if max_jll is not None else ''),
                "\t{}".format(self.model_params))
        elif it > self.n_burn_in_samples:
            print(
                "Chain {:>2}: {:>4}/{:d}\t{:.2f}s\t".format(self.chain_id, it - self.n_burn_in_samples, self.n_samples,
                                                            sample_time),
                "ARI: {:.3f}".format(metrics.adjusted_rand_score(self.Y, self._z)),
                "\t{:^25}".format('New max lik-log: {:.2f}'.format(max_jll) if max_jll is not None else ''),
                "\t{}".format(self.model_params))

        else:
            print(
                "*** Burn in period ***\t"
                "Chain {:>2}: {:>4}/{:d}\t{:.2f}s\t".format(self.chain_id, it, self.n_burn_in_samples,
                                                            sample_time),
                "ARI: {:.3f}".format(metrics.adjusted_rand_score(self.Y, self._z)),
                "\t{:^25}".format('New max lik-log: {:.2f}'.format(max_jll) if max_jll is not None else ''),
                "\t{}".format(self.model_params))

    def run(self, n_samples=10, n_prior_updates=20, n_burn_in_samples=5, starting_point=None, verbose=False,
            save_all_samples=False):
        if self.verbose > 0:
            print('Initiating chain {}...'.format(self.chain_id))
        self.save_all_samples = save_all_samples
        self.n_samples = n_samples
        self.n_burn_in_samples = n_burn_in_samples
        self.samples_ = np.zeros((n_samples + n_burn_in_samples + 1, self._n), dtype=np.int8)
        self.samples_log_lik_ = np.zeros((n_samples + n_burn_in_samples + 1, 1))
        self.n_prior_updates = n_prior_updates
        z = np.random.randint(self.C, size=(self._n,)) if starting_point is None else starting_point
        z = self.ensure_correct_z(z, raise_if_incorrect=starting_point is not None)
        self.set_z(z)
        self.verbose = verbose
        if verbose:
            self._numint_error = dict()

        self.samples_[0] = z.copy()
        self.samples_log_lik_[0] = self.log_lik() + self._log_joint_prior

        for key in self._infer_params:
            self.sampled_params_[key] = np.zeros((n_samples + n_burn_in_samples + 1, 1))

        prev_max_jll = self.samples_log_lik_[0]

        if self.verbose > 0:
            print('Initiated chain {}. Starting to sample...'.format(self.chain_id))

        for n in range(1, self.n_samples + self.n_burn_in_samples + 1):
            t0 = time.time()
            self._old_max_jll = None
            new_max_jll = False
            # metropolis-hastings sampling for hyperprior
            self.sample_prior(n_prior_samples=self.n_prior_updates)
            for key in self._infer_params:
                self.sampled_params_[key][n, 0] = self.model_params[key]

            np.random.shuffle(self._sample_indices)
            for i in self._sample_indices:
                max_jll = self.inner(i) + self._log_joint_prior

            if self.sort_communities:
                self._sort_communities()
            self.samples_[n, :] = self._z.copy()
            self.samples_log_lik_[n] = max_jll

            if max_jll > prev_max_jll:
                prev_max_jll = max_jll
                self.max_log_lik_z_ = self._z.copy()
                self.max_log_lik_ = max_jll
                self.max_log_lik_params_ = self.model_params.copy()
                new_max_jll = True

            # Saving the maximum log likelihood for the convergence plots
            self._max_log_list.append(max_jll)

            if verbose is not False and verbose > 1:
                self.print_progress(n, time.time() - t0, prev_max_jll if new_max_jll else None)

        self.set_z(self.max_log_lik_z_, self.max_log_lik_params_)
        # self.set_inferred_params(**deepcopy(self.max_log_lik_params_))
        if self.sort_communities:
            self._sort_communities()

        if self.verbose > 0:
            print('Finished sampling chain {}. Now starting deterministic optimization...'.format(self.chain_id))

        # anneal to the maximum
        n_anneal = 0
        self.anneal_max_log_lik_z_ = self._z.copy()
        self.anneal_max_log_lik_ = self.max_log_lik_
        self.anneal_max_log_lik_params_ = self.model_params.copy()
        while True:
            t0 = time.time()
            n_anneal += 1
            # sample_prior(n_prior_updates)
            improved = False
            np.random.shuffle(self._sample_indices)
            for i in self._sample_indices:
                max_jll = self.inner(i, False) + self._log_joint_prior
                if max_jll > prev_max_jll:
                    improved = True
                    self.anneal_max_log_lik_z_ = self._z.copy()
                    self.anneal_max_log_lik_ = max_jll
                    self.anneal_max_log_lik_params_ = self.model_params.copy()
                    prev_max_jll = max_jll

            if self.sort_communities:
                self._sort_communities()

            if not improved:
                break
            elif verbose is not False and verbose > 1:
                self.print_progress(n + n_anneal, time.time() - t0, max_jll)

        self.set_z(self.anneal_max_log_lik_z_, self.anneal_max_log_lik_params_)

        if self.verbose > 0:
            print('Finished deterministic optimization for chain {}.'.format(self.chain_id))

        # Create numpy objects for all samples
        if self.save_all_samples:
            self.px_z_all_ = np.array(self.px_z_all_)
            self.px_z_all_.reshape(-1, 1)
            self.z_all_ = np.array(self.z_all_)

        # Create separate objects for burn_in_objects and post_samples
        self.burn_in_samples_ = self.samples_[:self.n_burn_in_samples + 1]
        self.burn_in_samples_log_lik_ = self.samples_log_lik_[:self.n_burn_in_samples + 1]
        self.post_samples_ = self.samples_[self.n_burn_in_samples + 1:]
        self.post_samples_log_lik_ = self.samples_log_lik_[self.n_burn_in_samples + 1:]
        for key in self._infer_params:
            self.burn_in_sampled_params_[key] = self.sampled_params_[key][:self.n_burn_in_samples + 1]
            self.post_sampled_params_[key] = self.sampled_params_[key][self.n_burn_in_samples + 1:]

        if self.verbose > 0:
            print('Finished chain {}.'.format(self.chain_id))

        return self


class _BayesianCommunityDetection(_CommunityDetectionChain):
    """
    Base Class for Bayesian Models
    -------------------------------------------------

    Parameters
    ----------
    X : sparse scipy matrix, shape(number of nodes, number of nodes)
        The adjacency matrix

    model_params : dict
        Dictionary containing values for the parameter keys alpha_in, alpha_out, beta_in, beta_out, b and gamma.
        If a value for a key is None, the parameter is inferred.

    C : int, optional, default: 2
        The number of clusters to infer (current implementation only supports 2 clusters)

    Y : numpy array, optional
        Ground truth vector for the group assignments. This allows the sampler to report the Adjusted Random Index of
        new samples.

    shared_b : True
        Has to be True, because the model does not support b constraints

    block_sampling : boolean, optional, default: False
        All leaf nodes will be aggregated to a block and will not be considered for sampling

    marginalize_phi : boolean, optional, default: True
        Defines if the phi term of the likelihood function should be marginalized during calculation

    chain_id : int, optional, default: 1
        Chain identifier of the object

    Attributes
    ----------
    samples_ : numpy array, shape(number of samples + number of burn-in-samples, number of nodes)
        All samples from sampling

    samples_log_lik_ : numpy array, shape(number of samples + number of burn-in-samples, 1)
        Log-likelihood of each sample

    burn_in_samples_ : numpy array, shape(number of burn-in-samples, number of nodes)
        All samples from the burn-in-phase

    burn_in_samples_log_lik_ : numpy array, shape(number of burn-in-samples, number of nodes)
        Log-likelihood for each burn-in-sample

    post_samples_ : numpy array, shape(number of samples, number of nodes)
        All samples after burn-in-phase

    post_samples_log_lik_ : numpy array, shape(number of samples, 1)
        Log-likelihood for each sample after burn-in-phase

    anneal_max_log_lik_z_ : numpy array, shape(number of nodes,)
        Group assignment vector with highest log-likelihood obtained by deterministic optimization

    anneal_max_log_lik_ : float
        Log-likelihood for maximum log-likelihood group vector obtained by deterministic optimization

    anneal_max_log_lik_params_ : dict
        Dictionary containing the parameters for the group assignment vector with highest log-likelihood obtained by
        deterministic optimization

    sampled_params_ : dict
        Dictionary containing numpy arrays with shape(number of samples + number of burn-in-samples, 1) for each
        parameter inferred

    burn_in_sampled_params_ : dict
        Dictionary containing numpy arrays with shape(number of burn-in-samples, 1) for each parameter inferred

    post_sampled_params_ : dict
        Dictionary containing numpy arrays with shape(number of samples, 1) for each parameter inferred

    max_log_lik_z_ : numpy array, shape(number of nodes,)
        Group assignment vector with highest log-likelihood obtained by Gibbs sampler

    max_log_lik_ : float
        Log-likelihood for maximum log-likelihood group vector obtained by Gibbs sampler

    max_log_lik_params_ : dict
        Dictionary containing the parameters for the group assignment vector with highest log-likelihood obtained by
        Gibbs sampler

    z_all_ : numpy array, shape(number of total visited solutions, number of nodes)
        (only if the chain was run with save_all_samples=True)
        Matrix containing all samples for all solutions visited during sampling of this chain.

    px_z_all_ : numpy array, shape(number of total visited solutions, 1)
        (only if the chain was run with save_all_samples=True)
        Log-likelihood for each solution visited.
    """

    def __init__(self, X, model_params, C=2, Y=None, shared_b=False, block_sampling=False,
                 marginalize_phi=True, chain_id=1):
        self._log_C_eta = None
        self._min_alpha_beta = 1e-2
        infer_params = set()
        model_params = model_params.copy()
        params = {
            'alpha_in': self._min_alpha_beta + 1,
            'beta_in': self._min_alpha_beta + 1,
            'alpha_out': self._min_alpha_beta + 1,
            'beta_out': self._min_alpha_beta + 1
        }
        for p, v in params.items():
            if p not in model_params:
                model_params[p] = params[p]
            elif model_params[p] is None:
                model_params[p] = v
                infer_params.add(p)

        super().__init__(X, model_params, C, Y, shared_b, block_sampling, marginalize_phi, chain_id)

        self._infer_params = self._infer_params | infer_params

    def evaluate_integral(self, alpha_in, alpha_out, beta_in, beta_out):
        return 0.0

    def log_priors(self, param, x):
        param_type = param.split('_')[0]
        if param_type == 'alpha':
            return -np.log(x - self._min_alpha_beta)
        elif param_type == 'beta':
            return -np.log(x - self._min_alpha_beta)
        else:
            return super().log_priors(param, x)

    def jump(self, param_type, old_value):
        if param_type == 'alpha' or param_type == 'beta':
            return self._min_alpha_beta + np.exp(np.log(old_value - self._min_alpha_beta) + np.random.normal(0, 0.1)), 0
        else:
            return super().jump(param_type, old_value)

    def calculate_unconstrained_ml_etas(self):
        Nc = self._Nc + self.model_params['alpha_in']
        N_out = self._N_out + self.model_params['alpha_out']

        nc_pow2 = self._nc ** 2
        nc_pow2_sum = np.sum(nc_pow2)
        n_out_pow2 = self._n ** 2 - nc_pow2_sum + self.model_params['beta_out']
        nc_pow2_sum += self.model_params['beta_in']
        eta_in = 2 * Nc / nc_pow2
        eta_out = 2 * N_out / n_out_pow2

        return eta_in, eta_out

    def recalculate_constants_with_param(self, param):
        super().recalculate_constants_with_param(param)
        param_type = param.split('_')[0]
        if param_type in {'b', 'alpha', 'beta'} \
                and self._model_params_b is not None \
                and not np.isnan(self._model_params_b).any() \
                and None not in [self.model_params.get('alpha_in'), self.model_params.get('beta_in'),
                                 self.model_params.get('alpha_out'), self.model_params.get('beta_out')]:
            alpha_in = np.array([self.model_params['alpha_in']] * self.C)
            beta_in = np.array([self.model_params['beta_in']] * self.C)

            self._log_C_eta = -self.evaluate_integral(alpha_in, self.model_params['alpha_out'], beta_in,
                                                     self.model_params['beta_out'])


class _BayesianStochasticBlockmodelSharedEtaOut(_BayesianCommunityDetection):
    """
    Full bayesian graph cut without community constraint
    -------------------------------------------------

    Parameters
    ----------
    X : sparse scipy matrix, shape(number of nodes, number of nodes)
        The adjacency matrix

    model_params : dict
        Dictionary containing values for the parameter keys alpha_in, alpha_out, beta_in, beta_out, b and gamma.
        If a value for a key is None, the parameter is inferred.

    C : int, optional, default: 2
        The number of clusters to infer (current implementation only supports 2 clusters)

    Y : numpy array, optional
        Ground truth vector for the group assignments. This allows the sampler to report the Adjusted Random Index of
        new samples.

    shared_b : True
        Has to be True, because the model does not support b constraints

    block_sampling : boolean, optional, default: False
        All leaf nodes will be aggregated to a block and will not be considered for sampling

    marginalize_phi : boolean, optional, default: True
        Defines if the phi term of the likelihood function should be marginalized during calculation

    chain_id : int, optional, default: 1
        Chain identifier of the object

    Attributes
    ----------
    samples_ : numpy array, shape(number of samples + number of burn-in-samples, number of nodes)
        All samples from sampling

    samples_log_lik_ : numpy array, shape(number of samples + number of burn-in-samples, 1)
        Log-likelihood of each sample

    burn_in_samples_ : numpy array, shape(number of burn-in-samples, number of nodes)
        All samples from the burn-in-phase

    burn_in_samples_log_lik_ : numpy array, shape(number of burn-in-samples, number of nodes)
        Log-likelihood for each burn-in-sample

    post_samples_ : numpy array, shape(number of samples, number of nodes)
        All samples after burn-in-phase

    post_samples_log_lik_ : numpy array, shape(number of samples, 1)
        Log-likelihood for each sample after burn-in-phase

    anneal_max_log_lik_z_ : numpy array, shape(number of nodes,)
        Group assignment vector with highest log-likelihood obtained by deterministic optimization

    anneal_max_log_lik_ : float
        Log-likelihood for maximum log-likelihood group vector obtained by deterministic optimization

    anneal_max_log_lik_params_ : dict
        Dictionary containing the parameters for the group assignment vector with highest log-likelihood obtained by
        deterministic optimization

    sampled_params_ : dict
        Dictionary containing numpy arrays with shape(number of samples + number of burn-in-samples, 1) for each
        parameter inferred

    burn_in_sampled_params_ : dict
        Dictionary containing numpy arrays with shape(number of burn-in-samples, 1) for each parameter inferred

    post_sampled_params_ : dict
        Dictionary containing numpy arrays with shape(number of samples, 1) for each parameter inferred

    max_log_lik_z_ : numpy array, shape(number of nodes,)
        Group assignment vector with highest log-likelihood obtained by Gibbs sampler

    max_log_lik_ : float
        Log-likelihood for maximum log-likelihood group vector obtained by Gibbs sampler

    max_log_lik_params_ : dict
        Dictionary containing the parameters for the group assignment vector with highest log-likelihood obtained by
        Gibbs sampler

    z_all_ : numpy array, shape(number of total visited solutions, number of nodes)
        (only if the chain was run with save_all_samples=True)
        Matrix containing all samples for all solutions visited during sampling of this chain.

    px_z_all_ : numpy array, shape(number of total visited solutions, 1)
        (only if the chain was run with save_all_samples=True)
        Log-likelihood for each solution visited.
    """

    def __init__(self, X, model_params, C=2, Y=None, block_sampling=False,
                 marginalize_phi=True, chain_id=1):
        model_params['b'] = 1
        super().__init__(X, model_params, C, Y, shared_b=True, block_sampling=block_sampling,
                         marginalize_phi=marginalize_phi, chain_id=chain_id)
        self.sort_communities = False

    def evaluate_integral(self, alpha_in, alpha_out, beta_in, beta_out):
        return loggamma(alpha_out) - alpha_out * np.log(beta_out) \
               + np.sum(loggamma(alpha_in) - alpha_in * np.log(beta_in))

    def evaluate_eta_term(self):
        nc_pow2 = self._nc ** 2
        nc_pow2_sum = np.sum(nc_pow2)
        n_out_pow2 = self._n ** 2 - nc_pow2_sum

        alpha_out = self._N_out + self.model_params['alpha_out']
        beta_out = 0.5 * n_out_pow2 + self.model_params['beta_out']
        eta_post = loggamma(alpha_out) - alpha_out * np.log(beta_out)
        alpha_in = self._Nc + self.model_params['alpha_in']
        beta_in = self.model_params['beta_in'] + 0.5 * nc_pow2
        eta_post += np.sum(loggamma(alpha_in) - alpha_in * np.log(beta_in))

        return eta_post + self._log_C_eta


class _ShiftedApproximateBayesianCommunityDetection(_BayesianCommunityDetection):
    """
    Full bayesian graph cut with community constraint
    -------------------------------------------------

    Parameters
    ----------
    X : sparse scipy matrix, shape(number of nodes, number of nodes)
        The adjacency matrix

    model_params : dict
        Dictionary containing values for the parameter keys alpha_in, alpha_out, beta_in, beta_out, b and gamma.
        If a value for a key is None, the parameter is inferred.

    C : int, optional, default: 2
        The number of clusters to infer (current implementation only supports 2 clusters)

    Y : numpy array, optional
        Ground truth vector for the group assignments. This allows the sampler to report the Adjusted Random Index of
        new samples.

    shared_b : boolean, optional, default: False
        Defining if all clusters share the same b

    block_sampling : boolean, optional, default: False
        All leaf nodes will be aggregated to a block and will not be considered for sampling

    marginalize_phi : boolean, optional, default: True
        Defines if the phi term of the likelihood function should be marginalized during calculation

    chain_id : int, optional, default: 1
        Chain identifier of the object

    Attributes
    ----------
    saved_integrals_ : dict
        All unique calculated integrals.

    samples_ : numpy array, shape(number of samples + number of burn-in-samples, number of nodes)
        All samples from sampling

    samples_log_lik_ : numpy array, shape(number of samples + number of burn-in-samples, 1)
        Log-likelihood of each sample

    burn_in_samples_ : numpy array, shape(number of burn-in-samples, number of nodes)
        All samples from the burn-in-phase

    burn_in_samples_log_lik_ : numpy array, shape(number of burn-in-samples, number of nodes)
        Log-likelihood for each burn-in-sample

    post_samples_ : numpy array, shape(number of samples, number of nodes)
        All samples after burn-in-phase

    post_samples_log_lik_ : numpy array, shape(number of samples, 1)
        Log-likelihood for each sample after burn-in-phase

    anneal_max_log_lik_z_ : numpy array, shape(number of nodes,)
        Group assignment vector with highest log-likelihood obtained by deterministic optimization

    anneal_max_log_lik_ : float
        Log-likelihood for maximum log-likelihood group vector obtained by deterministic optimization

    anneal_max_log_lik_params_ : dict
        Dictionary containing the parameters for the group assignment vector with highest log-likelihood obtained by
        deterministic optimization

    sampled_params_ : dict
        Dictionary containing numpy arrays with shape(number of samples + number of burn-in-samples, 1) for each
        parameter inferred

    burn_in_sampled_params_ : dict
        Dictionary containing numpy arrays with shape(number of burn-in-samples, 1) for each parameter inferred

    post_sampled_params_ : dict
        Dictionary containing numpy arrays with shape(number of samples, 1) for each parameter inferred

    max_log_lik_z_ : numpy array, shape(number of nodes,)
        Group assignment vector with highest log-likelihood obtained by Gibbs sampler

    max_log_lik_ : float
        Log-likelihood for maximum log-likelihood group vector obtained by Gibbs sampler

    max_log_lik_params_ : dict
        Dictionary containing the parameters for the group assignment vector with highest log-likelihood obtained by
        Gibbs sampler

    z_all_ : numpy array, shape(number of total visited solutions, number of nodes)
        (only if the chain was run with save_all_samples=True)
        Matrix containing all samples for all solutions visited during sampling of this chain.

    px_z_all_ : numpy array, shape(number of total visited solutions, 1)
        (only if the chain was run with save_all_samples=True)
        Log-likelihood for each solution visited.
    """

    def __init__(self, X, model_params, C=2, Y=None, shared_b=False, block_sampling=False,
                 marginalize_phi=True, chain_id=1):
        super().__init__(X, model_params, C, Y, shared_b, block_sampling, marginalize_phi,
                         chain_id=chain_id)

    @lru_cache(maxsize=None)
    def pseudo_int_all_shifted_wo_fapprox_logdom3(self, alpha_in, alpha_out, beta_in, beta_out, b, K0, K1, K2):
        Bi = beta_in[1] / (b * beta_out)
        Ai = beta_in[0] / (b * beta_out)
        A = np.array([Ai, Bi])
        w_terms = np.zeros(shape=(2 * (2 * K0 - 1)))
        c = 0
        w_terms_max = -np.inf
        for (m, n) in [(0, 1), (1, 0)]:
            log_A0 = np.log(A[m])
            log_A1 = np.log(A[n])
            log_A2 = np.log(1 + A[m])
            log_A3 = np.log(1 + A[m] + A[n])
            aux = alpha_in[m] * log_A0 + alpha_in[n] * log_A1
            inner_sum_parts = np.array([-k * log_A2 + self.loggamma(alpha_out + alpha_in[m] + k)
                                        - self.loggamma(alpha_out + k + 1) for k in range(K0)])
            max_val = np.max(inner_sum_parts)
            lse = max_val + np.log(np.cumsum(np.exp(inner_sum_parts - max_val)))

            for w in range(2 * K0 - 1):
                w_term = w * log_A2 \
                         + self.loggamma(alpha_out + alpha_in[m] + alpha_in[n] + w) \
                         - self.loggamma(alpha_out + alpha_in[m] + w + 1) \
                         - log_A3 * (alpha_out + alpha_in[m] + alpha_in[n] + w) + aux \
                         + (lse[w] if w < K0 else lse[K0 - 1])
                if w_term > w_terms_max:
                    w_terms_max = w_term
                w_terms[c] = w_term
                c += 1

        integral = w_terms_max + np.log(np.sum(np.exp(w_terms - w_terms_max))) \
                   + self.loggamma(alpha_out) - np.sum(alpha_in * np.log(beta_in)) - alpha_out * np.log(beta_out)
        self.saved_integrals_[(*alpha_in, alpha_out, *beta_in, beta_out)] = integral
        return integral

    @lru_cache(maxsize=None)
    def loggamma(self, a):
        return loggamma(a)

    def logpoch(self, a, i):
        return self.loggamma(a + i) - self.loggamma(a)

    def pseudo_integral(self, alpha_in, alpha_out, beta_in, beta_out, b):
        """
        Evaluates integral of the form gamma pdf * gamma cdf

        Parameters
        ----------
        alpha_in : array[dims(2)]
        alpha_out : float
        beta_in : array[dims(2)]
        beta_out : float
        b : float



        Returns
        -------
        log_pseudoint : float
            value of the integral
        """

        # suppressing warnings for low levels of verbosity
        warnings.filterwarnings("ignore") if (self.verbose < 3) else warnings.filterwarnings("default")
        log_eta_norm = 0
        log_eta_norm += np.sum(alpha_in * np.log(beta_in) - loggamma(alpha_in))

        f = lambda x: np.exp(-x) * (x ** (alpha_out - 1)) * gammaincc(alpha_in[0],
                                                                      x * beta_in[0] / (b * beta_out)) * gammaincc(
            alpha_in[1], x * beta_in[1] / (b * beta_out)) * gamma(alpha_in[0]) * gamma(alpha_in[1])

        pars = (tuple(alpha_in), alpha_out, tuple(beta_in), beta_out, b)
        if self.verbose is not False and self.verbose > 2:
            print('chain:{0},\n parameters {{alpha_in,alpha_out,beta_in,beta_out,b }}:\n {1}\n'
                  .format(self.chain_id, (tuple(alpha_in), alpha_out, tuple(beta_in), beta_out, b)))

        maxK = +2e6  # maximum feasible value for gamma function evaluation
        qq = 0.9999

        K1p = int(
            np.min([maxK, gamma_dist.ppf(qq, a=np.nanmax([alpha_in]), scale=1 / np.nanmin([beta_in])) * beta_out]) + 1)
        if (K1p < maxK):
            K2p = K1p
            K0p = 0
            log_pseudoint = self.pseudo_int_all_shifted_wo_fapprox_logdom3(*pars, K0=K1p, K1=0, K2=0)
            pseudoint = np.exp(log_pseudoint)
        else:
            warnings.warn("The derived K exceeded 2e6. Due to runtime reasons K is limitted to 2e6. Be aware that this "
                          "means that the error bounding guarantee is not satisfied anymore.", Warning)
            K1p = int(2e6)
            K2p = K1p
            K0p = 0
            log_pseudoint = self.pseudo_int_all_shifted_wo_fapprox_logdom3(*pars, K0=K1p, K1=0, K2=0)
            pseudoint = np.exp(log_pseudoint)

        if self.verbose is not False and self.verbose > 2:
            exact_int_v = pseudoint
            log_exact_int = log_pseudoint
            warnings.filterwarnings("error")

            try:
                (exact_int_v, exact_int_v_error) = quad(f, 0, np.inf)  # / (gamma(alpha_in[0]) * gamma(alpha_in[1]) * gamma(alpha_out))
                exact_int_v = float(exact_int_v)
                log_exact_int = np.log(exact_int_v) - np.sum(alpha_in * np.log(beta_in)) - alpha_out * np.log(beta_out)
            except Warning as e:
                if e.__class__.__name__ == 'IntegrationWarning':
                    print('Integration warning, numerical integral disregarded. Params:{0}'
                          .format((tuple(alpha_in), alpha_out, tuple(beta_in), beta_out, b)))
                    exact_int_v = pseudoint
                    log_exact_int = log_pseudoint
                    raise ArithmeticError
                else:
                    pass

            print('Added obs: \t{0},\n pseudo integral: \t{1},\n exact_integral: \t{2} \n ratio of exps:{3}'
                  .format(K1p, log_pseudoint, log_exact_int, (float(pseudoint) - np.exp(log_exact_int)) /
                          np.exp(log_exact_int)))
            # tracking numerical integral vs. pseudo integral difference
            self._numint_error[pars] = ((float(pseudoint) - exact_int_v) / exact_int_v)
            # switching warnings into default mode of first occurence being printed

            warnings.filterwarnings("default")

        return float(log_pseudoint)

    def evaluate_integral(self, alpha_in, alpha_out, beta_in, beta_out):
        log_integral = self.pseudo_integral(alpha_in, alpha_out, beta_in, beta_out,
                                            b=self.model_params.get('b') or self.model_params.get('b_1'))

        return log_integral

    def evaluate_eta_term(self):
        nc_pow2 = self._nc ** 2
        nc_pow2_sum = np.sum(nc_pow2)
        n_out_pow2 = self._n ** 2 - nc_pow2_sum
        alpha_out = self._N_out + self.model_params['alpha_out']
        beta_out = 0.5 * n_out_pow2 + self.model_params['beta_out']
        alpha_in = self._Nc + self.model_params['alpha_in']
        beta_in = 0.5 * nc_pow2 + self.model_params['beta_in']
        result = self.evaluate_integral(alpha_in, alpha_out, beta_in, beta_out)
        return result + self._log_C_eta


def _unwrap_run_function(arg, **kwarg):
    """
    Wrapping function to enable parallel computing by the joblib package

    Parameters
    ----------
    arg : list
        All mandatory arguments required for defining a class
    kwarg : dict
        Dictionary containing possible keyword arguments for the class to be defined

    Returns
    -------
    chain : object
        Chain after sampling containing all samples
    """
    return _CommunityDetectionChain.run(*arg, **kwarg)


class Model(object):
    """
    Bayesian model for performing graph cuts
    ----------------------------------------

    - **ShiftedApproximateBayesianCommunityDetection**: Full bayesian graph cut with community constraint
    - **BayesianStochasticBlockmodelSharedEtaOut**: Full bayesian graph cut without community constraint

    Parameters
    ----------

    Cut : string {'ShiftedApproximateBayesianCommunityDetection', 'BayesianStochasticBlockmodelSharedEtaOut'}
        The model for cutting the graph

    X : sparse scipy matrix, shape(n, n)
        The (:math:`nxn`) adjacency matrix

    model_params : dict
        Dictionary containing values for the parameter keys alpha_in, alpha_out, beta_in, beta_out, b and gamma.
        If a value for a key is None, the parameter is inferred.

    C : int, optional, default: 2
        The number of clusters to infer (current implementation only supports 2 clusters)

    Y : numpy array, optional
        Ground truth vector for the group assignments. This allows the sampler to report the Adjusted Random Index of
        new samples.

    block_sampling : boolean, optional, default: False
        All leaf nodes will be aggregated to a block and will not be considered for sampling

    marginalize_phi : boolean, optional, default: True
        Defines if the phi term of the likelihood function should be marginalized during calculation

    Attributes
    ----------
    chains : list
        List of chains (derived classes of the model)
    """

    def __init__(self, Cut, X, model_params, C=2, Y=None, block_sampling=False,
                 marginalize_phi=False):
        self.Cut = Cut
        self.args = [X, model_params]
        if C > 2:
            raise NotImplementedError('This package does currently not support C higher than 2')
        self.kwargs = {
            'Y': Y,
            'C': C}
        if block_sampling:
            self.kwargs['block_sampling'] = block_sampling
        if marginalize_phi:
            self.kwargs['marginalize_phi'] = marginalize_phi
        self.chains = []

    def add_chains(self, number_of_chains=1):
        """
        Adds chains to the model that can be used for inference

        Parameters
        ----------
        number_of_chains : int, optional, default: 1

        """

        for n in range(number_of_chains):
            self.chains.append(globals()['_' + self.Cut](*self.args, **self.kwargs,
                                                   chain_id=len(self.chains) + 1))

    def run_chains(self, n_samples=10, n_prior_updates=20, n_burn_in_samples=5, starting_point=None, verbose=False,
                   save_all_samples=False, parallel=True):
        """
        Runs all chains of the Model with the specified parameters

        Parameters
        ----------
        n_samples : int, optional, default: 10
            Number of samples to be obtained by the sampler

        n_prior_updates : int, optional, default: 20
            Number of prior sweeps for each sample

        n_burn_in_samples : int, optional, default: 5
            Number of burn-in-samples

        starting_point: numpy array, optional, default: None
            Set the starting sample for the chain

        verbose: boolean or int, optional, default: False
            Setting the verbosity level of reports. There are several levels with following meaning (higher levels
            include lower ones)
            True or 1: Print basic information about the status of the chain, e.g. start sampling, finish sampling etc.
            2: Print information about each sample
            3: (only effective for ShiftedApproximateBayesianCommunityDetection) print all parameters and calculate the
            error of the approximation (This will significantly slow down the speed of the calculations).

        save_all_samples : boolean, optional, default: False
            Allows saving all samples, also the not accepted ones

        parallel : boolean, optional, default: True
            Defines if parallel computation should be used for sampling.
        """

        self.save_all_samples = save_all_samples
        if parallel:
            self.chains = Parallel(n_jobs=-1, max_nbytes='100M') \
                (delayed(_unwrap_run_function)((chain, n_samples, n_prior_updates, n_burn_in_samples, starting_point,
                                               verbose, save_all_samples)) for chain in self.chains)
        else:
            for chain in self.chains:
                chain.run(n_samples, n_prior_updates, n_burn_in_samples, starting_point, verbose, save_all_samples)

    def get_best_chain(self):
        """
        Return the chain with the best sample in terms of maximum log likelihood

        Returns
        -------
        chain : object
            The chain of the model with the best sample in terms of maximum log likelihood
        """

        chain_max_log = []
        # highest likelihood
        for chain in self.chains:
            chain_max_log.append(chain.max_log_lik_)

        best_chain_index = np.argmax(chain_max_log)
        return self.chains[best_chain_index]

    def aggregate_chain_samples(self, with_burn_in=False):
        """
        Return all samples from all chains and return

        Parameters
        ----------
        with_burn_in : boolean, optional, default: False
            Whether to include the burn-in-samples or not

        Returns
        -------
        z_matrix_post_samples : numpy array, shape(number of total samples, number of nodes)
            Matrix containing all samples from all chains.

        log_prob_post_samples : numpy array, shape(number of total samples, 1)
            Log-likelihood for each sample.

        z_matrix_random_samples : numpy array, shape(number of total visited solutions, number of nodes)
            (only if the chains were run with save_all_samples=True)
            Matrix containing all samples for all solutions visited during sampling of all chains.

        log_prob_random_samples : numpy array, shape(number of total visited solutions, 1)
            (only if the chains were run with save_all_samples=True)
            Log-likelihood for each solution visited.
        """

        z_matrix_post_samples = None
        log_prob_post_samples = None
        z_matrix_random_samples = None
        log_prob_random_samples = None

        for chain in self.chains:
            # Running over post samples
            if z_matrix_post_samples is None:
                if with_burn_in:
                    z_matrix_post_samples = chain.samples_
                else:
                    z_matrix_post_samples = chain.post_samples_
            else:
                if with_burn_in:
                    z_matrix_post_samples = np.vstack((z_matrix_post_samples, chain.samples_))
                else:
                    z_matrix_post_samples = np.vstack((z_matrix_post_samples, chain.post_samples_))
            if log_prob_post_samples is None:
                if with_burn_in:
                    log_prob_post_samples = chain.samples_log_lik_
                else:
                    log_prob_post_samples = chain.post_samples_log_lik_
            else:
                if with_burn_in:
                    log_prob_post_samples = np.vstack((log_prob_post_samples, chain.samples_log_lik_))
                else:
                    log_prob_post_samples = np.vstack((log_prob_post_samples, chain.post_samples_log_lik_))
            # Running over all samples
            if self.save_all_samples:
                if z_matrix_random_samples is None:
                    z_matrix_random_samples = chain.z_all_
                else:
                    z_matrix_random_samples = np.vstack((z_matrix_random_samples, chain.z_all_))
                if log_prob_random_samples is None:
                    log_prob_random_samples = chain.px_z_all_
                else:
                    log_prob_random_samples = np.hstack((log_prob_random_samples, chain.px_z_all_))

        return z_matrix_post_samples, log_prob_post_samples, z_matrix_random_samples, log_prob_random_samples
