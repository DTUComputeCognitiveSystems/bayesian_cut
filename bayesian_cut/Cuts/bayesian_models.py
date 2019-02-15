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

EPS = 1e-200


class CommunityDetectionChain(object):
    """
    :param X:
    :param model_params:
    :param C:
    :param Y:
    :param shared_eta_in:
    :param shared_b:
    :param block_sampling:
    :param marginalize_phi:
    :param chain_id:
    """

    def __init__(self, X, model_params, C=2, Y=None, shared_eta_in=False, shared_b=False, block_sampling=False,
                 marginalize_phi=True, chain_id=1):
        self.C = C
        self.marginalize_phi = marginalize_phi
        self.shared_eta_in = shared_eta_in
        self.shared_b = shared_b
        self.sort_communities = True

        self.saved_integrals = {}
        self.X = X.copy()
        self.X_blocks = self.X.copy()
        self.k = np.squeeze(np.asarray(X.sum(axis=1)))
        self.n = X.shape[0]
        self.N = int(np.sum(self.k) / 2)
        self.self_links = self.X.diagonal() / 2  # keep true self-link count in a separate array
        self.X.setdiag(0)  # don't store self link in the main matrix

        # faster way to access data from adjacency matrix
        self.X_indices = {}
        self.X_data = {}
        for i in range(X.shape[0]):
            row = self.X[i]
            self.X_indices[i] = row.indices
            self.X_data[i] = row.data

        if block_sampling:
            self.sample_indices = np.where((self.X > 0).sum(1) > 1)[0]
        else:
            self.sample_indices = list(range(self.n))
        self.X_blocks[:, self.sample_indices] = 0
        self.X_blocks.eliminate_zeros()
        self.X_blocks_sums = self.X_blocks.copy()
        self.X_blocks_sums.data = self.self_links[self.X_blocks_sums.indices]
        self.X_blocks_self_links_sum = self.X_blocks_sums.sum(1).A1
        self.X_blocks_sums = self.X_blocks_self_links_sum + self.X_blocks.sum(1).A1 + self.self_links
        self.X_blocks.setdiag(1)
        X_blocks_dict = {}
        for i in self.sample_indices:
            X_blocks_dict[i] = self.X_blocks[i].indices
        self.X_blocks = X_blocks_dict

        self.Y = Y if Y is not None else np.zeros((self.n,), dtype=np.int)

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

        self.z = None
        self.max_log_list = []
        self.Nc = None
        self.N_out = None
        self.nc = None
        self.Kc = None

        if not self.marginalize_phi:
            self.k_log_k = np.sum(self.k * np.log(self.k + 1))  # will crash if there are nodes with no links

        self.chain_id = chain_id
        self.log_C_phi = 0
        self.old_max_jll_ = -np.inf
        self.anneal_max_log_lik_z = None
        self.anneal_max_log_lik = None
        self.anneal_max_log_lik_params = None
        self.max_log_lik_z = None
        self.max_log_lik = None
        self.max_log_lik_params = None

        # Allow saving of all samples when required
        self.px_z_all_ = []
        self.z_all_ = []

        # Global verbose level
        self.verbose = False

        # Tracking of numerical errors
        self.numint_error = None

        self.model_params = {}
        self.model_params_b = None if self.shared_b else np.array([np.nan] * self.C)  # collects bs into a list
        self.infer_params = set()
        self.sampled_params_ = {}
        self.burn_in_sampled_params_ = {}
        self.post_sampled_params_ = {}
        self.log_joint_prior = 0

        if model_params['gamma'] is None:
            model_params['gamma'] = 1.0
            self.infer_params.add('gamma')
        if model_params['b'] is None:
            model_params['b'] = 0.999
            if self.shared_b:
                self.infer_params.add('b')
            else:
                for i in range(1, C + 1):
                    self.infer_params.add('b_{}'.format(i))

        for pname, value in model_params.items():
            if pname != 'b' or self.shared_b:
                self.set_param(pname, value, False)
            else:
                for i in range(1, C + 1):
                    self.set_param('b_{}'.format(i), value, False)
        self.calc_joint_prior()

    def log_lik(self):
        if (self.nc > 0).sum() < self.C:
            return -np.inf

        return np.sum(self.log_lik_terms())

    def log_lik_terms(self):
        eta_term = self.evaluate_eta_term()
        phi_term = np.sum(self.Kc * np.log(self.nc))

        if self.marginalize_phi:
            phi_term += np.sum(loggamma(self.nc * self.model_params['gamma'])) \
                        - np.sum(loggamma(self.Kc + self.nc * self.model_params['gamma'])) \
                        + self.log_C_phi
        else:
            phi_term += -np.sum(self.Kc * np.log(self.Kc)) + self.k_log_k

        return eta_term, phi_term

    def evaluate_eta_term(self):
        pass

    def calc_log_lik_for_z(self, z, joint_with_prior=False):
        self.set_z(z)
        return self.log_lik() + (self.log_joint_prior if joint_with_prior else 0)

    def calculate_unconstrained_ml_etas(self):
        nc_pow2 = self.nc ** 2
        nc_pow2_sum = np.sum(nc_pow2)
        n_out_pow2 = self.n ** 2 - nc_pow2_sum
        if self.shared_eta_in:
            N_in = np.sum(self.Nc)
            eta_in = 2 * N_in / nc_pow2_sum
            eta_out = 2 * self.N_out / n_out_pow2
        else:
            eta_in = 2 * self.Nc / nc_pow2
            eta_out = 2 * self.N_out / n_out_pow2

        return eta_in, eta_out

    def set_z(self, z, params=None):
        self.z = z.copy()
        self._calc_node_assignments()
        self.calc_links_nodes()
        if params is not None:
            for pname, value in params.items():
                self.set_param(pname, value)
            self.calc_joint_prior()

    def ensure_correct_z(self, z, raise_if_incorrect=False):
        for i in self.sample_indices:
            if raise_if_incorrect and not np.all(z[self.X_blocks[i]] == z[i]):
                raise Exception('Incorrect starting point.')
            else:
                z[self.X_blocks[i]] = z[i]

        return z

    def update_links_nodes_block(self, i, from_group, to_group):
        from_group_links, to_group_links = self._node_aux[[from_group, to_group], i]
        update_node_aux_indices = self.X_indices[i]
        update_node_aux_data = self.X_data[i]
        block_size = self.X_blocks[i].size
        self.nc[to_group] += block_size
        self.nc[from_group] -= block_size
        self.Kc[to_group] += self.k[i] + self.X_blocks_self_links_sum[i] + self.X_blocks_sums[i]
        self.Kc[from_group] -= self.k[i] + self.X_blocks_self_links_sum[i] + self.X_blocks_sums[i]
        self.Nc[to_group] += to_group_links + self.X_blocks_sums[i]
        self.Nc[from_group] -= from_group_links + self.X_blocks_sums[i]
        self.N_out += from_group_links - to_group_links
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
        self.Nc = np.zeros(shape=(self.C,))
        self.nc = np.zeros(shape=(self.C,))
        self.Kc = np.zeros(shape=(self.C,))
        for c in range(self.C):
            idx = np.where(self.z == c)[0]
            self.Kc[c] = self.k[idx].sum()
            self.Nc[c] = self.X[idx, :].tocsc()[:, idx].sum() / 2 + self.self_links[idx].sum()
            self.nc[c] = idx.size

        self.N_out = self.N - self.Nc.sum()

    def _calc_node_assignments(self):
        aux = np.zeros((self.n,), dtype=np.bool)
        aux[self.sample_indices] = True
        self._node_aux = np.zeros(shape=(self.C, self.n))
        for c in range(self.C):
            idx = np.where((self.z == c) & aux)[0]
            self._node_aux[c] = self.X[idx, :].sum(axis=0)

    def _sort_communities(self):
        eta_in, _ = self.calculate_unconstrained_ml_etas()
        order = (-eta_in).argsort()
        map_z = dict(zip(order, np.arange(self.C)))
        if np.any(np.diff(order) < 0):
            self.Nc = self.Nc[order]
            self.nc = self.nc[order]
            self.Kc = self.Kc[order]
            self._node_aux = self._node_aux[order]
            self.z = np.vectorize(map_z.__getitem__)(self.z)

    def inner(self, i, stochastic=True):
        init_assign = self.z[i]
        px_z = np.array([-np.inf] * self.C)
        px_z[init_assign] = self.log_lik() if self.old_max_jll_ is None else self.old_max_jll_
        if self.save_all_samples:
            self.px_z_all_.append(px_z[init_assign])
            self.z_all_.append(self.z.copy())

        for k in range(self.C):
            if k != init_assign:
                self.update_links_nodes_block(i, self.z[i], k)
                self.z[self.X_blocks[i]] = k
                px_z[k] = self.log_lik()
                if self.save_all_samples:
                    # Add samples to memory
                    self.z_all_.append(self.z.copy())
                    # Add corresponding probability to memory
                    self.px_z_all_.append(px_z[k])

        max_ll = px_z.copy()
        if stochastic:
            px_z = np.exp(px_z - np.max(px_z))
            px_z /= np.sum(px_z)
            update = np.random.choice(self.C, p=px_z)
        else:
            update = np.argmax(px_z)

        if self.z[i] != update:
            self.update_links_nodes_block(i, self.z[i], update)
            self.z[self.X_blocks[i]] = update

        self.old_max_jll_ = max_ll[update]

        return max_ll[update]

    def log_priors(self, param, x):
        param_type = param.split('_')[0]
        if param_type == 'gamma':
            return -np.log(x)
        elif param_type == 'b':
            return -np.log(x)

    def calc_joint_prior(self):
        self.log_joint_prior = np.sum([self.log_priors(pname, self.model_params[pname]) for pname in self.infer_params])

    def set_param(self, param, value, update_joint_prior=True):
        if value != self.model_params.get(param):
            self.model_params[param] = value
            self.recalculate_constants_with_param(param)
            if update_joint_prior:
                self.calc_joint_prior()

    def recalculate_constants_with_param(self, param):
        param_type = param.split('_')[0]
        if param_type == 'gamma' and self.marginalize_phi:
            self.log_C_phi = np.sum(loggamma(self.k + self.model_params['gamma'])) - \
                             self.n * loggamma(self.model_params['gamma'])
        elif param_type == 'b':
            if self.shared_b:
                self.model_params_b = self.model_params['b']
            else:
                self.model_params_b[int(param.split('_')[-1]) - 1] = self.model_params[param]

    def jump(self, param_type, old_value):
        beta_n = 200
        if param_type == 'gamma':
            new_value = np.clip(np.exp(np.log(old_value) + np.random.normal(0, 0.1)), 0, 10 * self.n)
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
        if len(self.infer_params) == 0:
            return
        prev_log_lik = None
        for n in range(n_prior_samples):
            for pname in self.infer_params:
                old_value = self.model_params[pname]
                old_log_lik = prev_log_lik if prev_log_lik is not None else self.log_lik()
                new_value, correction = self.jump(pname.split('_')[0], old_value)
                # TODO: temporary hack, to be removed later?
                if pname == 'alpha_in' and new_value > self.model_params['beta_in']:
                    new_value = old_value
                elif pname == 'alpha_out' and new_value > self.model_params['beta_out']:
                    new_value = old_value
                elif pname == 'beta_in' and new_value < self.model_params['alpha_in']:
                    new_value = old_value
                elif pname == 'beta_out' and new_value < self.model_params['alpha_out']:
                    new_value = old_value
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
                "ARI: {:.3f}".format(metrics.adjusted_rand_score(self.Y, self.z)),
                "\t{:^25}".format('New max lik-log: {:.2f}'.format(max_jll) if max_jll is not None else ''),
                "\t{}".format(self.model_params))
        elif it > self.n_burn_in_samples:
            print(
                "Chain {:>2}: {:>4}/{:d}\t{:.2f}s\t".format(self.chain_id, it - self.n_burn_in_samples, self.n_samples,
                                                            sample_time),
                "ARI: {:.3f}".format(metrics.adjusted_rand_score(self.Y, self.z)),
                "\t{:^25}".format('New max lik-log: {:.2f}'.format(max_jll) if max_jll is not None else ''),
                "\t{}".format(self.model_params))

        else:
            print(
                "*** Burn in period ***\t"
                "Chain {:>2}: {:>4}/{:d}\t{:.2f}s\t".format(self.chain_id, it, self.n_burn_in_samples,
                                                            sample_time),
                "ARI: {:.3f}".format(metrics.adjusted_rand_score(self.Y, self.z)),
                "\t{:^25}".format('New max lik-log: {:.2f}'.format(max_jll) if max_jll is not None else ''),
                "\t{}".format(self.model_params))

    def run(self, n_samples=10, n_prior_updates=20, n_burn_in_samples=5, starting_point=None, verbose=False,
            save_all_samples=False):
        self.save_all_samples = save_all_samples
        self.n_samples = n_samples
        self.n_burn_in_samples = n_burn_in_samples
        self.samples_ = np.zeros((n_samples + n_burn_in_samples + 1, self.n), dtype=np.int8)
        self.samples_log_lik_ = np.zeros((n_samples + n_burn_in_samples + 1, 1))
        self.n_prior_updates = n_prior_updates
        z = np.random.randint(self.C, size=(self.n,)) if starting_point is None else starting_point
        z = self.ensure_correct_z(z, raise_if_incorrect=starting_point is not None)
        self.set_z(z)
        self.verbose = verbose
        if verbose:
            self.numint_error = dict()

        self.samples_[0] = z.copy()
        self.samples_log_lik_[0] = self.log_lik() + self.log_joint_prior

        for key in self.infer_params:
            self.sampled_params_[key] = np.zeros((n_samples + n_burn_in_samples + 1, 1))

        prev_max_jll = self.samples_log_lik_[0]

        for n in range(1, self.n_samples + self.n_burn_in_samples + 1):
            t0 = time.time()
            self.old_max_jll_ = None
            new_max_jll = False
            # metropolis-hastings sampling for hyperprior
            self.sample_prior()
            for key in self.infer_params:
                self.sampled_params_[key][n, 0] = self.model_params[key]

            np.random.shuffle(self.sample_indices)
            for i in self.sample_indices:
                max_jll = self.inner(i) + self.log_joint_prior

            if self.sort_communities:
                self._sort_communities()
            self.samples_[n, :] = self.z.copy()
            self.samples_log_lik_[n] = max_jll

            if max_jll > prev_max_jll:
                prev_max_jll = max_jll
                self.max_log_lik_z = self.z.copy()
                self.max_log_lik = max_jll
                self.max_log_lik_params = self.model_params.copy()
                new_max_jll = True

            # Saving the maximum log likelihood for the convergence plots
            self.max_log_list.append(max_jll)

            if verbose is not False and verbose > 1:
                self.print_progress(n, time.time() - t0, prev_max_jll if new_max_jll else None)

        self.set_z(self.max_log_lik_z, self.max_log_lik_params)
        # self.set_inferred_params(**deepcopy(self.max_log_lik_params))
        if self.sort_communities:
            self._sort_communities()

        # anneal to the maximum
        n_anneal = 0
        self.anneal_max_log_lik_z = self.z.copy()
        self.anneal_max_log_lik = self.max_log_lik
        self.anneal_max_log_lik_params = self.model_params.copy()
        while True:
            t0 = time.time()
            n_anneal += 1
            # sample_prior(n_prior_updates)
            improved = False
            np.random.shuffle(self.sample_indices)
            for i in self.sample_indices:
                max_jll = self.inner(i, False) + self.log_joint_prior
                if max_jll > prev_max_jll:
                    improved = True
                    self.anneal_max_log_lik_z = self.z.copy()
                    self.anneal_max_log_lik = max_jll
                    self.anneal_max_log_lik_params = self.model_params.copy()
                    prev_max_jll = max_jll

            if self.sort_communities:
                self._sort_communities()

            if not improved:
                break
            elif verbose is not False and verbose > 1:
                self.print_progress(n + n_anneal, time.time() - t0, max_jll)

        self.set_z(self.anneal_max_log_lik_z, self.anneal_max_log_lik_params)

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
        for key in self.infer_params:
            self.burn_in_sampled_params_[key] = self.sampled_params_[key][:self.n_burn_in_samples + 1]
            self.post_sampled_params_[key] = self.sampled_params_[key][self.n_burn_in_samples + 1:]

        return self


class BayesianCommunityDetection(CommunityDetectionChain):
    def __init__(self, X, model_params, C=2, Y=None, shared_eta_in=False, shared_b=False, block_sampling=False,
                 marginalize_phi=True, chain_id=1):
        self.log_C_eta = None
        self.min_alpha_beta = 1e-2
        infer_params = set()
        model_params = model_params.copy()
        params = {
            'alpha_in': self.min_alpha_beta + 1,
            'beta_in': self.min_alpha_beta + 1,
            'alpha_out': self.min_alpha_beta + 1,
            'beta_out': self.min_alpha_beta + 1
        }
        for p, v in params.items():
            if p not in model_params:
                model_params[p] = params[p]
            elif model_params[p] is None:
                model_params[p] = v
                infer_params.add(p)

        super().__init__(X, model_params, C, Y, shared_eta_in, shared_b, block_sampling, marginalize_phi, chain_id)

        self.infer_params = self.infer_params | infer_params

    def evaluate_integral(self, alpha_in, alpha_out, beta_in, beta_out):
        return 0.0

    def log_priors(self, param, x):
        param_type = param.split('_')[0]
        if param_type == 'alpha':
            return -np.log(x - self.min_alpha_beta)
        elif param_type == 'beta':
            return -np.log(x - self.min_alpha_beta)
        else:
            return super().log_priors(param, x)

    def jump(self, param_type, old_value):
        if param_type == 'alpha' or param_type == 'beta':
            return self.min_alpha_beta + np.exp(np.log(old_value - self.min_alpha_beta) + np.random.normal(0, 0.1)), 0
        else:
            return super().jump(param_type, old_value)

    def calculate_unconstrained_ml_etas(self):
        Nc = self.Nc + self.model_params['alpha_in']
        N_out = self.N_out + self.model_params['alpha_out']

        nc_pow2 = self.nc ** 2
        nc_pow2_sum = np.sum(nc_pow2)
        n_out_pow2 = self.n ** 2 - nc_pow2_sum + self.model_params['beta_out']
        nc_pow2_sum += self.model_params['beta_in']

        if self.shared_eta_in:
            N_in = np.sum(Nc)
            eta_in = 2 * N_in / nc_pow2_sum
            eta_out = 2 * N_out / n_out_pow2
        else:
            eta_in = 2 * Nc / nc_pow2
            eta_out = 2 * N_out / n_out_pow2

        return eta_in, eta_out

    def recalculate_constants_with_param(self, param):
        super().recalculate_constants_with_param(param)
        param_type = param.split('_')[0]
        if param_type in {'b', 'alpha', 'beta'} \
                and self.model_params_b is not None \
                and not np.isnan(self.model_params_b).any() \
                and None not in [self.model_params.get('alpha_in'), self.model_params.get('beta_in'),
                                 self.model_params.get('alpha_out'), self.model_params.get('beta_out')]:
            alpha_in = self.model_params['alpha_in'] if self.shared_eta_in else \
                np.array([self.model_params['alpha_in']] * self.C)
            beta_in = self.model_params['beta_in'] if self.shared_eta_in else \
                np.array([self.model_params['beta_in']] * self.C)

            self.log_C_eta = -self.evaluate_integral(alpha_in, self.model_params['alpha_out'], beta_in,
                                                     self.model_params['beta_out'])


class BayesianStochasticBlockmodelSharedEtaOut(BayesianCommunityDetection):
    def __init__(self, X, model_params, C=2, Y=None, shared_eta_in=False, block_sampling=False,
                 marginalize_phi=True, chain_id=1):
        model_params['b'] = 1
        super().__init__(X, model_params, C, Y, shared_eta_in, shared_b=True, block_sampling=block_sampling,
                         marginalize_phi=marginalize_phi, chain_id=chain_id)
        self.sort_communities = False

    # TODO implement shared_eta_in, reuse code with evaluate eta_term
    def evaluate_integral(self, alpha_in, alpha_out, beta_in, beta_out):
        return loggamma(alpha_out) - alpha_out * np.log(beta_out) \
               + np.sum(loggamma(alpha_in) - alpha_in * np.log(beta_in))

    def evaluate_eta_term(self):
        nc_pow2 = self.nc ** 2
        nc_pow2_sum = np.sum(nc_pow2)
        n_out_pow2 = self.n ** 2 - nc_pow2_sum

        alpha_out = self.N_out + self.model_params['alpha_out']
        beta_out = 0.5 * n_out_pow2 + self.model_params['beta_out']
        eta_post = loggamma(alpha_out) - alpha_out * np.log(beta_out)
        if self.shared_eta_in:
            alpha_in = np.sum(self.Nc) + self.model_params['alpha_in']
            beta_in = 0.5 * nc_pow2_sum + self.model_params['beta_in']
            eta_post += loggamma(alpha_in) - alpha_in * np.log(beta_in)
        else:
            alpha_in = self.Nc + self.model_params['alpha_in']
            beta_in = self.model_params['beta_in'] + 0.5 * nc_pow2
            eta_post += np.sum(loggamma(alpha_in) - alpha_in * np.log(beta_in))

        return eta_post + self.log_C_eta


class ShiftedApproximateBayesianCommunityDetection(BayesianCommunityDetection):
    def __init__(self, X, model_params, C=2, Y=None, shared_eta_in=False, shared_b=False, block_sampling=False,
                 marginalize_phi=True, chain_id=1):
        super().__init__(X, model_params, C, Y, shared_eta_in, shared_b, block_sampling, marginalize_phi,
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
        self.saved_integrals[(*alpha_in, alpha_out, *beta_in, beta_out)] = integral
        return integral

    @lru_cache(maxsize=None)
    def loggamma(self, a):
        return loggamma(a)

    def logpoch(self, a, i):
        return self.loggamma(a + i) - self.loggamma(a)

    def pseudo_integral(self, alpha_in, alpha_out, beta_in, beta_out, b):
        '''Evaluates integral of the form gamma pdf * gamma cdf
        ;inputs: alpha_in: array[dims(2)], alpha_out: float, K: summation boundary
        :returns (float) value of the integral
        '''

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
            self.numint_error[pars] = ((float(pseudoint) - exact_int_v) / exact_int_v)
            # switching warnings into default mode of first occurence being printed

            warnings.filterwarnings("default")

        return float(log_pseudoint)

    def evaluate_integral(self, alpha_in, alpha_out, beta_in, beta_out):
        log_integral = self.pseudo_integral(alpha_in, alpha_out, beta_in, beta_out,
                                            b=self.model_params.get('b') or self.model_params.get('b_1'))

        return log_integral

    def evaluate_eta_term(self):
        nc_pow2 = self.nc ** 2
        nc_pow2_sum = np.sum(nc_pow2)
        n_out_pow2 = self.n ** 2 - nc_pow2_sum

        alpha_out = self.N_out + self.model_params['alpha_out']
        beta_out = 0.5 * n_out_pow2 + self.model_params['beta_out']

        if self.shared_eta_in:
            alpha_in = np.sum(self.Nc) + self.model_params['alpha_in']
            beta_in = 0.5 * nc_pow2_sum + self.model_params['beta_in']
        else:
            alpha_in = self.Nc + self.model_params['alpha_in']
            beta_in = 0.5 * nc_pow2 + self.model_params['beta_in']
        result = self.evaluate_integral(alpha_in, alpha_out, beta_in, beta_out)
        return result + self.log_C_eta


def unwrap_run_function(arg, **kwarg):
    return CommunityDetectionChain.run(*arg, **kwarg)


class Model(object):
    def __init__(self, Cut, X, model_params, C=2, Y=None, shared_eta_in=None, shared_b=None, block_sampling=None,
                 marginalize_phi=None):
        self.Cut = Cut
        self.args = [X, model_params]
        self.kwargs = {
            'Y': Y,
            'C': C}
        if shared_eta_in is not None:
            self.kwargs['shared_eta_in'] = shared_eta_in
        if shared_b is not None:
            self.kwargs['shared_b'] = shared_b
        if block_sampling is not None:
            self.kwargs['block_sampling'] = block_sampling
        if marginalize_phi is not None:
            self.kwargs['marginalize_phi'] = marginalize_phi
        self.chains = []

    def add_chains(self, number_of_chains=1):
        for n in range(number_of_chains):
            self.chains.append(globals()[self.Cut](*self.args, **self.kwargs,
                                                   chain_id=len(self.chains) + 1))

    def run_chains(self, n_samples=10, n_prior_updates=20, n_burn_in_samples=5, starting_point=None, verbose=False,
                   save_all_samples=False, parallel=True):
        self.save_all_samples = save_all_samples
        if parallel:
            self.chains = Parallel(n_jobs=-1, max_nbytes='100M') \
                (delayed(unwrap_run_function)((chain, n_samples, n_prior_updates, n_burn_in_samples, starting_point,
                                               verbose, save_all_samples)) for chain in self.chains)
        else:
            for chain in self.chains:
                chain.run(n_samples, n_prior_updates, n_burn_in_samples, starting_point, verbose, save_all_samples)

    def get_best_chain(self):
        # Find the best chain and return it
        chain_max_log = []
        # highest likelihood
        for chain in self.chains:
            chain_max_log.append(chain.max_log_lik)

        best_chain_index = np.argmax(chain_max_log)
        return self.chains[best_chain_index]

    def aggregate_chain_samples(self, with_burn_in=False):
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
