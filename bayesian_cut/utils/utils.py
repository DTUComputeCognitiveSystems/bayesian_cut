#!/usr/bin/env python3
#
# -*- coding: utf-8 -*-
#
# Author:   Laurent Vermue <lauve@dtu.dk>
#           Maciej Korzepa <mjko@dtu.dk>
#           Petr Taborsky <ptab@dtu.dk>
#
# License: 3-clause BSD

import collections
from collections import defaultdict
import matplotlib.pyplot as plt
import naturalneighbor
import networkx as nx
import numpy as np
import plotly
import plotly.graph_objs as go
import scipy.sparse as sp
import seaborn as sb
from joblib import Parallel, delayed
from matplotlib import patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
from scipy.sparse.csgraph import connected_components
from scipy.stats import bayes_mvs
from sklearn.manifold import MDS
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
from bayesian_cut.cuts.bayesian_models import Model


def generate_synthetic_network(n=[250, 250], d=[0.2, 0.2], d_out=0.01):
    """
    Generates a synthetic network consisting of a number of clusters with d link densities and shared d_out density
    between these clusters. The number of clusters is defined by the length of input arrays n and d.
    :param n: an array of cluster sizes
    :param d: an array of link densities within clusters
    :param d_out: density of links between any two clusters
    :return: 2D sparse adjacency matrix, array of nodes' cluster ids
    """
    total_n = np.sum(n)
    Y = np.zeros(shape=(total_n,))
    A = (np.random.random(size=(total_n, total_n)) < d_out).astype(np.int8)
    C = len(n)
    curr = 0
    for c in range(C):
        Y[curr:curr + n[c]] = c
        A[curr:curr + n[c], curr:curr + n[c]] = (np.random.random(size=(n[c], n[c])) < d[c]).astype(np.int8)
        curr += n[c]
    A = np.tril(A)
    A = A + A.T
    np.fill_diagonal(A, 0)
    return sp.csr_matrix(A, dtype=np.int8), Y


def generate_synthetic_network_homogeneous_degrees(n=[250, 250], d=[0.2, 0.2], d_out=0.01):
    """
    Same as generate_synthetic_network, but ensures that all nodes have roughly same degree.
    """
    total_n = np.sum(n)
    Y = np.zeros(shape=(total_n,))
    A = np.zeros((total_n, total_n))
    C = len(n)
    # Add outside links with homogeneous degree
    curr = 0
    for c in range(C - 1):
        curr_col = 0
        for g in range(0, C):
            if c < g:
                rows = n[c]
                columns = n[g]
                approx_links = np.ceil(d_out * np.min((rows, columns)))
                total_links = approx_links * np.max((rows, columns))
                max_links = approx_links * (np.max((rows, columns)) / np.min((rows, columns)))
                approx_degree = (rows * approx_links) / columns
                print('The density of links between cluster {:d} and cluster {:d} will be {:.2f} with {:.0f} links'.
                      format(c + 1, g + 1, total_links / (rows * columns), total_links))
                for row in range(curr, curr + n[c]):
                    for column in range(curr_col, curr_col + n[g]):
                        if rows >= columns:
                            if np.sum(A[row, curr_col:(curr_col + n[g])]) < approx_links \
                                    and np.sum(A[curr:(curr + n[c]), column]) < max_links:
                                A[row, column] = 1
                        else:
                            if np.sum(A[row, curr_col:(curr_col + n[g])]) < max_links \
                                    and np.sum(A[curr:(curr + n[c]), column]) < approx_links:
                                A[row, column] = 1
            curr_col += n[g]
        curr += n[c]
    A = A.T
    # Add inside links with homogeneous degrees
    curr = 0
    for c in range(C):
        Y[curr:curr + n[c]] = c
        fills = np.ceil(d[c] * (n[c] - 1) / 2)
        if fills * 2 > (n[c] - 1):
            approx_links = fills * 2 - 1
        else:
            approx_links = fills * 2
        approx_degree = approx_links / (n[c] - 1)
        print('The degree is calculated without self-links')
        print('The degree of each node in cluster {:d} will be {:.2f} with {:.0f} links'.format(c + 1, approx_degree,
                                                                                                approx_links))
        for k in range(1, int(fills) + 1):
            np.fill_diagonal(A[k + curr:(curr + n[c]), (curr):(k + curr + n[c])], 1)
            np.fill_diagonal(A[(curr + n[c] - k):(curr + n[c]), (curr):(curr + n[c] - k)], 1)

        curr += n[c]
    # Build final Adjacency matrix
    A = np.tril(A)
    A = A + A.T
    np.fill_diagonal(A, 0)
    return sp.csr_matrix(A, dtype=np.int8), Y


def landscape_plotting(sim_matrix, log_prob_matrix, z_solutions=None, z_matrix=None, draw_contours=False,
                       title='Solution landscape', z_title='z', filename='landscape_plot.html', gaussian_sigma=None,
                       open_plot=False, res=500j, verbose=False):
    if z_solutions is not None and z_matrix is None:
        raise AssertionError('Given solutions cannot be plotted without providing the z_matrix, which was used to'
                             'create the similarity matrix.')
    # Obtain grid size of the data
    xmin = sim_matrix[:, 0].min()
    xmax = sim_matrix[:, 0].max()
    ymin = sim_matrix[:, 1].min()
    ymax = sim_matrix[:, 1].max()
    XX, YY = np.mgrid[xmin:xmax:res, ymin:ymax:res]

    grid_ranges = [[xmin, xmax, res], [ymin, ymax, res], [0, 1, 1]]

    sim_matrix_new = np.hstack((sim_matrix, np.zeros((sim_matrix.shape[0], 1))))

    Z_cut = griddata(sim_matrix, log_prob_matrix, (XX, YY), method='linear', fill_value=0)
    Z_cut = Z_cut.squeeze()
    # Perform grid interpolation of the actual likelihood values and the created grid
    Z = naturalneighbor.griddata(sim_matrix_new, np.squeeze(log_prob_matrix), grid_ranges)
    Z = Z.squeeze()
    # Smoothen the data with Gaussian Kernel
    if gaussian_sigma is not None:
        Z = gaussian_filter(Z, sigma=gaussian_sigma)
    XX[Z_cut == 0] = np.nan
    YY[Z_cut == 0] = np.nan
    Z[Z_cut == 0] = np.nan
    if verbose:
        print('Maximum value found for Landscape', np.nanmax(Z))
        print('Maximum value found in log vector', np.nanmax(log_prob_matrix))

    contours = {}
    if draw_contours:
        contours['contours'] = go.surface.Contours(
            z=go.surface.contours.Z(
                show=True,
                usecolormap=True,
                highlightcolor="#42f462",
                project=dict(z=True)
            )
        )
    data = [
        go.Surface(
            x=XX,
            y=YY,
            z=Z,
            showscale=False,
            **contours
        )
    ]

    layout = go.Layout(
        showlegend=False,
        autosize=False,
        title=title,
        scene=dict(
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=1),
            zaxis=dict(
                ticks='',
                title=z_title,
                # titlefont=dict(
                #     size=22,
                #     color='black'
                # ),
                # tickfont=dict(
                #     size=21,
                #     color='black'
                # ),
            ),
            xaxis=dict(
                ticks='',
                title='',
                showticklabels=False,
            ),
            yaxis=dict(
                ticks='',
                title='',
                showticklabels=False
            )
        ),
        width=500,
        height=500,
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        )
    )

    fig = go.Figure(data=data, layout=layout)

    # Plot Trace
    if z_solutions is not None:
        if not isinstance(z_solutions, list):
            z_solutions = [z_solutions]
        # Approximate Coordinates
        num_samples = len(z_solutions)
        for i in range(num_samples):
            vec_index = np.where(np.all(z_solutions[i] == z_matrix, axis=1))[0][0]
            x_index = np.unravel_index(np.nanargmin(np.abs(XX - sim_matrix[vec_index][0])), XX.shape)[0]
            y_index = np.unravel_index(np.nanargmin(np.abs(YY - sim_matrix[vec_index][1])), YY.shape)[1]
            approx_x = XX[x_index, y_index]
            approx_y = YY[x_index, y_index]
            approx_z = Z[x_index, y_index]
            trace = dict(type='scatter3d',
                         x=[approx_x],
                         y=[approx_y],
                         z=[approx_z],
                         mode='markers+text',
                         line=dict(color='red'),
                         marker=dict(size=10, color='yellow', symbol='circle', line=dict(color='black', width=2))
                         )
            fig.add_trace(trace)

    plotly.offline.plot(fig, filename=filename, auto_open=open_plot, config={'showSendToCloud': True, 'showLink': True})


def landscape_plotting_notebook(sim_matrix, log_prob_matrix, z_solutions=None, z_matrix=None, draw_contours=False,
                       title='Solution landscape', z_title='z', gaussian_sigma=None, res=500j, verbose=False):
    if z_solutions is not None and z_matrix is None:
        raise AssertionError('Given solutions cannot be plotted without providing the z_matrix, which was used to'
                             'create the similarity matrix.')
    # Obtain grid size of the data
    xmin = sim_matrix[:, 0].min()
    xmax = sim_matrix[:, 0].max()
    ymin = sim_matrix[:, 1].min()
    ymax = sim_matrix[:, 1].max()
    XX, YY = np.mgrid[xmin:xmax:res, ymin:ymax:res]

    grid_ranges = [[xmin, xmax, res], [ymin, ymax, res], [0, 1, 1]]

    sim_matrix_new = np.hstack((sim_matrix, np.zeros((sim_matrix.shape[0], 1))))

    Z_cut = griddata(sim_matrix, log_prob_matrix, (XX, YY), method='linear', fill_value=0)
    Z_cut = Z_cut.squeeze()
    # Perform grid interpolation of the actual likelihood values and the created grid
    Z = naturalneighbor.griddata(sim_matrix_new, np.squeeze(log_prob_matrix), grid_ranges)
    Z = Z.squeeze()
    # Smoothen the data with Gaussian Kernel
    if gaussian_sigma is not None:
        Z = gaussian_filter(Z, sigma=gaussian_sigma)
    XX[Z_cut == 0] = np.nan
    YY[Z_cut == 0] = np.nan
    Z[Z_cut == 0] = np.nan
    if verbose:
        print('Maximum value found for Landscape', np.nanmax(Z))
        print('Maximum value found in log vector', np.nanmax(log_prob_matrix))

    contours = {}
    if draw_contours:
        contours['contours'] = go.surface.Contours(
            z=go.surface.contours.Z(
                show=True,
                usecolormap=True,
                highlightcolor="#42f462",
                project=dict(z=True)
            )
        )
    data = [
        go.Surface(
            x=XX,
            y=YY,
            z=Z,
            showscale=False,
            **contours
        )
    ]

    layout = go.Layout(
        showlegend=False,
        autosize=False,
        title=title,
        scene=dict(
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=1),
            zaxis=dict(
                ticks='',
                title=z_title,
                # titlefont=dict(
                #     size=22,
                #     color='black'
                # ),
                # tickfont=dict(
                #     size=21,
                #     color='black'
                # ),
            ),
            xaxis=dict(
                ticks='',
                title='',
                showticklabels=False,
            ),
            yaxis=dict(
                ticks='',
                title='',
                showticklabels=False
            )
        ),
        width=500,
        height=500,
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        )
    )

    fig = go.Figure(data=data, layout=layout)

    # Plot Trace
    if z_solutions is not None:
        if not isinstance(z_solutions, list):
            z_solutions = [z_solutions]
        # Approximate Coordinates
        num_samples = len(z_solutions)
        for i in range(num_samples):
            vec_index = np.where(np.all(z_solutions[i] == z_matrix, axis=1))[0][0]
            x_index = np.unravel_index(np.nanargmin(np.abs(XX - sim_matrix[vec_index][0])), XX.shape)[0]
            y_index = np.unravel_index(np.nanargmin(np.abs(YY - sim_matrix[vec_index][1])), YY.shape)[1]
            approx_x = XX[x_index, y_index]
            approx_y = YY[x_index, y_index]
            approx_z = Z[x_index, y_index]
            trace = dict(type='scatter3d',
                         x=[approx_x],
                         y=[approx_y],
                         z=[approx_z],
                         mode='markers+text',
                         line=dict(color='red'),
                         marker=dict(size=10, color='yellow', symbol='circle', line=dict(color='black', width=2))
                         )
            fig.add_trace(trace)

    plotly.offline.iplot(fig)


def trace_plot(model, show_burn_in=False, model_names=None, title=None, legend=False):
    plt.figure(figsize=(10, 6))
    color_list = ['b', 'r', 'g', 'c', 'm', 'k', 'y']
    if isinstance(model, list):
        plottitle = ''
        for i, model in enumerate(model):
            for chain in model.chains:
                if show_burn_in:
                    plt.plot(chain.samples_log_lik_, color=color_list[i], alpha=0.4, label='{}'
                             .format(model_names[i] if chain.chain_id == 1 else ''))
                else:
                    plt.plot(chain.post_samples_log_lik_, color=color_list[i], alpha=0.4, label='{}'
                             .format(model_names[i] if chain.chain_id == 1 else ''))
            plottitle += model.Cut + '\n' if model_names is None else model_names[i] + '\n'

        if title:
            plt.title(title)
        elif isinstance(title, str):
            plt.title('Trace-plot for \n{}{}'.format(plottitle, '\nBurn in samples included' if show_burn_in else ''),
                      fontsize=19)
    else:
        for chain in model.chains:
            if show_burn_in:
                plt.plot(chain.samples_log_lik_, label="Chain {:d}".format(chain.chain_id))
            else:
                plt.plot(chain.post_samples_log_lik_, label="Chain {:d}".format(chain.chain_id))
        if isinstance(title, str):
            plt.title(title)
        elif title:
            plt.title('Trace-plot for {}{}'.format(model.Cut, '\n\nBurn in samples included' if show_burn_in else ''),
                      fontsize=19)

    if legend:
        plt.legend(loc=(1.04, 0))
    plt.ylabel('log-likelihood')
    plt.xlabel('number of samples')
    plt.tight_layout()
    plt.show()


def nmi_score_overall(model):
    n_chains = len(model.chains)
    convergence_vec = np.zeros(int(((n_chains * n_chains) - n_chains) / 2))
    chain_index, subchain_index = np.triu_indices(n_chains, 1)
    run = 0
    for chain, subchain in zip(chain_index, subchain_index):
        NMI_result = normalized_mutual_info_score(model.chains[chain].max_log_lik_z_,
                                                  model.chains[subchain].max_log_lik_z_)
        convergence_vec[run] = NMI_result
        run = run + 1

    mean_NMI, _, _ = bayes_mvs(convergence_vec, alpha=0.95)
    if np.isnan(mean_NMI[0]):
        print('There is no variation in the data. No confidence intervals will be given')
        return np.mean(convergence_vec)
    else:
        return mean_NMI[0], mean_NMI[1][0], mean_NMI[1][1]


def calc_modularity_scores(z_matrix, adjacency_matrix):
    if len(z_matrix.shape) == 1:
        z_matrix = z_matrix.reshape(1, -1)
    adjacency_matrix = adjacency_matrix.todense()
    np.fill_diagonal(adjacency_matrix, 0)
    num_samples = z_matrix.shape[0]
    modularity_scores = np.zeros((num_samples, 1))
    for i in range(num_samples):
        z_index_zero = z_matrix[i] == 0
        z_index_one = ~z_index_zero
        degree_vec = np.sum(adjacency_matrix, axis=1)
        m = np.sum(degree_vec) / 2
        deduct_matrix = degree_vec.dot(degree_vec.T) / (2 * m)
        B = adjacency_matrix - deduct_matrix
        z_copy = z_matrix[i].copy()
        z_copy[z_index_zero] = -1
        z_copy[z_index_one] = 1
        modularity_scores[i, 0] = (1/(4*m)) * z_copy.T.dot(B).dot(z_copy)
    return modularity_scores


def calc_ratiocut_scores(z_matrix, adjacency_matrix):
    if len(z_matrix.shape) == 1:
        z_matrix = z_matrix.reshape(1, -1)
    adjacency_matrix = adjacency_matrix.todense()
    np.fill_diagonal(adjacency_matrix, 0)
    num_samples = z_matrix.shape[0]
    ratiocut_scores = np.zeros((num_samples, 1))
    for i in range(num_samples):
        groups = np.unique(z_matrix[i])
        cost = 0
        for group in groups:
            group_index = z_matrix[i] == group
            num_vertices = group_index.sum()
            weight_cut = (adjacency_matrix[group_index, :][:, ~group_index]).sum()
            cost = cost + (weight_cut / num_vertices)
        cost = cost / 2
        ratiocut_scores[i, 0] = cost
    return ratiocut_scores


def calc_normcut_scores(z_matrix, adjacency_matrix):
    if len(z_matrix.shape) == 1:
        z_matrix = z_matrix.reshape(1, -1)
    adjacency_matrix = adjacency_matrix.todense()
    np.fill_diagonal(adjacency_matrix, 0)
    num_samples = z_matrix.shape[0]
    normcut_scores = np.zeros((num_samples, 1))
    for i in range(num_samples):
        groups = np.unique(z_matrix[i])
        cost = 0
        for group in groups:
            group_index = z_matrix[i] == group
            weight_cut = (adjacency_matrix[group_index, :][:, ~group_index]).sum()
            edge_weight = (adjacency_matrix[group_index, :].sum() - weight_cut).sum() / 2 + weight_cut
            cost = cost + (weight_cut / edge_weight)
        cost = cost / 2
        normcut_scores[i, 0] = cost
    return normcut_scores


def param_plot(model, include_burn_in=False):
    params = model.chains[0].infer_params
    for param in params:
        plt.figure()
        for chain in model.chains:
            if include_burn_in:
                sb.kdeplot(np.squeeze(chain.sampled_params_[param]), label='Chain_{}'.format(chain.chain_id))
            else:
                sb.kdeplot(np.squeeze(chain.post_sampled_params_[param]), label='Chain_{}'.format(chain.chain_id))
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        if include_burn_in:
            plt.title("\n{:s} parameter samples\nMethod: {:s}\nChains: {:d} Samples per chain: {:d}{}\n"
                      .format(param, model.Cut, len(model.chains), model.chains[0].samples_log_lik_.shape[0],
                              '(including burn in)'))
        else:
            plt.title("\n{:s} parameter samples\nMethod: {:s}\nChains: {:d} Samples per chain: {:d}\n"
                      .format(param, model.Cut, len(model.chains), model.chains[0].n_samples))
        if len(model.chains) > 10:
            plt.legend().set_visible(False)
        plt.tight_layout()
        plt.show()


def cluster_plot(X, z_vector=None, ground_truth=None, colorbar=False, title='Adjacency matrix'):
    if isinstance(X, Model):
        z_vector = X.get_best_chain().max_log_lik_z_
        X = X.args[0]

    def hcluster_plot(X, r_list, start=0, stop=None, use_gray=True):

        def add_cut(i, start, stop, color='b', linestyle='-', linewidth=1.5):
            eps = 0.5
            plt.plot([start - eps, stop - eps], [i - eps, i - eps], color=color, linestyle=linestyle,
                     linewidth=linewidth)
            plt.plot([i - eps, i - eps], [start - eps, stop - eps], color=color, linestyle=linestyle,
                     linewidth=linewidth)

        stop = stop or X.shape[0]

        if isinstance(r_list[0], collections.Iterable):
            l_list = [len(_flatten(r_list[i])) for i in range(len(r_list))]
        else:
            # Writing inner density
            X[start:stop, start:stop] = np.mean(X[start:stop, start:stop])
            return
        covered_index_row = 0
        for i, l in enumerate(l_list):
            add_cut(start + l + covered_index_row, start, stop)
            if use_gray:
                covered_index_col = 0
                for j in range(len(l_list)):
                    if j > i:
                        # Writing upper angle density
                        X[start + covered_index_row:start + covered_index_row + l,
                        start + covered_index_col:start + covered_index_col + l_list[j]] = \
                            np.mean(X[start + covered_index_row:start + covered_index_row + l,
                                    start + covered_index_col:start + covered_index_col + l_list[j]])
                        # Writing lower triangle density
                        X[start + covered_index_col:start + covered_index_col + l_list[j],
                        start + covered_index_row:start + covered_index_row + l] = \
                            np.mean(X[start + covered_index_col:start + covered_index_col + l_list[j],
                                    start + covered_index_row:start + covered_index_row + l])
                        # Next in between cluster
                    covered_index_col += l_list[j]
            # Next cluster
            covered_index_row += l

        covered_index = 0
        for i, l in enumerate(l_list):
            hcluster_plot(X, r_list[i], start + covered_index, start + covered_index + l, use_gray=use_gray)
            covered_index += l

    Y = ground_truth
    X_new = X.toarray()
    if z_vector is None:
        print('Due to the missing z_vector no cut can be drawn in the adjaceny matrix')
        z_vector = np.zeros(X.shape[0])
    r_list = _assignment_array_to_lists(z_vector)
    X_ordered, _ = _reorder_adjacency_matrix(X_new, r_list, plot_diag=False)

    points = X_ordered.copy()

    fig, ax = plt.subplots(figsize=(8, 8))
    plt.xticks([])
    plt.yticks([])

    hcluster_plot(X_ordered, r_list)

    X_ordered = X_ordered + points
    final_img = np.minimum(X_ordered, 1)
    plt.imshow(final_img, cmap='hot_r', vmax=1, vmin=0)
    plt.title(title, fontsize=17)

    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = plt.colorbar(cax=cax)
        cb.ax.tick_params(labelsize=10)

    if Y is not None:
        ax = plt.gca()
        tick_classes = [Y[i] for list in r_list for i in list]
        ax.set_xticks(np.arange(0, len(tick_classes), 1))
        ax.set_yticks(np.arange(0, len(tick_classes), 1))
        ax.set_xticklabels(tick_classes)
        ax.set_yticklabels(tick_classes)
        x_labels = len(tick_classes) * ['|']
        y_labels = len(tick_classes) * ['—']
        for num, group in enumerate(tick_classes):
            if group == 0:
                ax.get_xticklabels()[num].set_color('b')
                ax.get_yticklabels()[num].set_color('b')
            else:
                ax.get_xticklabels()[num].set_color('r')
                ax.get_yticklabels()[num].set_color('r')
        ax.set_xticklabels(x_labels)
        ax.set_yticklabels(y_labels)
    plt.tight_layout()
    plt.show()


def heatmap_plot(sim_matrix, log_prob_matrix, chain_traces=None, z_matrix=None, title='Title', gaussian_sigma=None,
                 filename='heatmap_plot.html', open_plot=False):
    if chain_traces is not None and z_matrix is None:
        raise AssertionError('Chain traces cannot be plotted without providing the z_matrix, which was used to create'
                             'the similarity matrix.')
    # Obtain grid size of the data
    xmin = sim_matrix[:, 0].min()
    xmax = sim_matrix[:, 0].max()
    ymin = sim_matrix[:, 1].min()
    ymax = sim_matrix[:, 1].max()
    XX, YY = np.mgrid[xmin:xmax:600j, ymin:ymax:600j]

    grid_ranges = [[xmin, xmax, 600j], [ymin, ymax, 600j], [0, 1, 1]]

    sim_matrix_new = np.hstack((sim_matrix, np.zeros((sim_matrix.shape[0], 1))))

    Z_cut = griddata(sim_matrix, log_prob_matrix, (XX, YY), method='linear', fill_value=0)
    Z_cut = Z_cut.squeeze()
    # Perform grid interpolation of the actual likelihood values and the created grid
    Z = naturalneighbor.griddata(sim_matrix_new, np.squeeze(log_prob_matrix), grid_ranges)
    Z = Z.squeeze()
    # Smoothen the data with Gaussian Kernel
    if gaussian_sigma is not None:
        Z = gaussian_filter(Z, sigma=gaussian_sigma)
    # XX[Z_cut == 0] = np.nan
    # YY[Z_cut == 0] = np.nan
    Z[Z_cut == 0] = np.nan

    data = [
        go.Heatmap(
            z=Z.T,
            x=np.linspace(xmin, xmax, 600),
            y=np.linspace(ymin, ymax, 600),
        )
    ]

    layout = go.Layout(
        title=title,
        xaxis=dict(ticks='', nticks=36),
        yaxis=dict(ticks=''),
        width=800,
        height=800,
        showlegend=False,
        margin=dict(
            l=65,
            r=50,
            b=65,
            t=90
        )
    )

    fig = go.Figure(data=data, layout=layout)

    # Plot Trace
    if chain_traces is not None:
        color_list = ['blue', 'red', 'green', 'cyan', 'magenta', 'black', 'brown']
        for j, trace in enumerate(chain_traces):
            num_samples = trace.shape[0]
            approx_x = np.zeros(num_samples)
            approx_y = np.zeros(num_samples)
            for i in range(num_samples):
                vec_index = np.where(np.all(trace[i] == z_matrix, axis=1))[0][0]
                x_index = np.unravel_index(np.nanargmin(np.abs(XX - sim_matrix[vec_index][0])), XX.shape)[0]
                y_index = np.unravel_index(np.nanargmin(np.abs(YY - sim_matrix[vec_index][1])), YY.shape)[1]
                approx_x[i] = XX[x_index, y_index]
                approx_y[i] = YY[x_index, y_index]

            if j < len(color_list):
                color = color_list[j]
            else:
                color = "rgb({}, {}, {})".format(*np.random.randint(256, size=3))
            trace = dict(type='scatter',
                         x=approx_x,
                         y=approx_y,
                         mode='markers+lines',
                         line=dict(color=color),
                         marker=dict(size=10, color=color, symbol='circle')
                         )

            fig.add_trace(trace)

    plotly.offline.plot(fig, filename=filename, auto_open=open_plot)


def heatmap_plot_notebook(sim_matrix, log_prob_matrix, chain_traces=None, z_matrix=None, title='Heatmap Trace Plot',
                          gaussian_sigma=None):
    if chain_traces is not None and z_matrix is None:
        raise AssertionError('Chain traces cannot be plotted without providing the z_matrix, which was used to create'
                             'the similarity matrix.')
    # Obtain grid size of the data
    xmin = sim_matrix[:, 0].min()
    xmax = sim_matrix[:, 0].max()
    ymin = sim_matrix[:, 1].min()
    ymax = sim_matrix[:, 1].max()
    XX, YY = np.mgrid[xmin:xmax:600j, ymin:ymax:600j]

    grid_ranges = [[xmin, xmax, 600j], [ymin, ymax, 600j], [0, 1, 1]]

    sim_matrix_new = np.hstack((sim_matrix, np.zeros((sim_matrix.shape[0], 1))))

    Z_cut = griddata(sim_matrix, log_prob_matrix, (XX, YY), method='linear', fill_value=0)
    Z_cut = Z_cut.squeeze()
    # Perform grid interpolation of the actual likelihood values and the created grid
    Z = naturalneighbor.griddata(sim_matrix_new, np.squeeze(log_prob_matrix), grid_ranges)
    Z = Z.squeeze()
    # Smoothen the data with Gaussian Kernel
    if gaussian_sigma is not None:
        Z = gaussian_filter(Z, sigma=gaussian_sigma)
    # XX[Z_cut == 0] = np.nan
    # YY[Z_cut == 0] = np.nan
    Z[Z_cut == 0] = np.nan

    data = [
        go.Heatmap(
            z=Z.T,
            x=np.linspace(xmin, xmax, 600),
            y=np.linspace(ymin, ymax, 600),
        )
    ]

    layout = go.Layout(
        title=title,
        xaxis=dict(ticks='', nticks=36),
        yaxis=dict(ticks=''),
        width=1200,
        height=1200,
        showlegend=False,
        margin=dict(
            l=65,
            r=50,
            b=65,
            t=90
        )
    )

    fig = go.Figure(data=data, layout=layout)

    # Plot Trace
    if chain_traces is not None:
        color_list = ['blue', 'red', 'green', 'cyan', 'magenta', 'black', 'brown']
        for j, trace in enumerate(chain_traces):
            num_samples = trace.shape[0]
            approx_x = np.zeros(num_samples)
            approx_y = np.zeros(num_samples)
            for i in range(num_samples):
                vec_index = np.where(np.all(trace[i] == z_matrix, axis=1))[0][0]
                x_index = np.unravel_index(np.nanargmin(np.abs(XX - sim_matrix[vec_index][0])), XX.shape)[0]
                y_index = np.unravel_index(np.nanargmin(np.abs(YY - sim_matrix[vec_index][1])), YY.shape)[1]
                approx_x[i] = XX[x_index, y_index]
                approx_y[i] = YY[x_index, y_index]

            if j < len(color_list):
                color = color_list[j]
            else:
                color = "rgb({}, {}, {})".format(*np.random.randint(256, size=3))
            trace = dict(type='scatter',
                         x=approx_x,
                         y=approx_y,
                         mode='markers+lines',
                         line=dict(color=color),
                         marker=dict(size=10, color=color, symbol='circle')
                         )

            fig.add_trace(trace)

    plotly.offline.iplot(fig)


def generate_new_samples(num_of_samples, model, unique_z_matrix, unique_log_prob):
    number_of_vectors = unique_log_prob.shape[0]
    number_of_nodes = unique_z_matrix.shape[1]
    new_z_matrix = np.zeros((num_of_samples, number_of_nodes))
    new_log_prob = np.zeros((num_of_samples, 1))
    for i in range(num_of_samples):
        a, b = np.random.choice(number_of_vectors, size=2, replace=False)
        bool_index = np.random.choice([True, False], size=number_of_nodes)
        new_z_matrix[i] = unique_z_matrix[a].copy()
        new_z_matrix[i][bool_index] = unique_z_matrix[b][bool_index]
        new_log_prob[i, 0] = model.chains[0].calc_log_lik_for_z(new_z_matrix[i])

    return new_z_matrix, new_log_prob


def generate_new_samples_only(num_of_samples, unique_z_matrix):
    number_of_vectors = unique_z_matrix.shape[0]
    number_of_nodes = unique_z_matrix.shape[1]
    new_z_matrix = np.zeros((num_of_samples, number_of_nodes))
    for i in range(num_of_samples):
        a, b = np.random.choice(number_of_vectors, size=2, replace=False)
        bool_index = np.random.choice([True, False], size=number_of_nodes)
        new_z_matrix[i] = unique_z_matrix[a].copy()
        new_z_matrix[i][bool_index] = unique_z_matrix[b][bool_index]

    return new_z_matrix


def recalculate_prob_for_z(new_model, z_matrix, verbose=False):
    number_of_samples = z_matrix.shape[0]
    new_log_prob_matrix = np.zeros((number_of_samples, 1))
    if len(new_model.chains) == 0:
        new_model.add_chains()
    for i in range(number_of_samples):
        if verbose:
            print('Model: {} Recalculating number: {}'.format(new_model.Cut, i + 1))
        new_log_prob_matrix[i, 0] = new_model.chains[-1].calc_log_lik_for_z(z_matrix[i])

    return new_log_prob_matrix


def calc_sim_matrix(z_matrix):
    number_of_vectors = z_matrix.shape[0]

    # Compute the variation of information between all z_vectors
    iterator = np.vstack(np.triu_indices(number_of_vectors, 1)).T
    sim_matrix = np.zeros((number_of_vectors, number_of_vectors))

    results = Parallel(n_jobs=-1, max_nbytes='100M') \
        (delayed(_variation_of_information)(z_matrix[row], z_matrix[column]) for row, column in iterator)

    for i, row_col in enumerate(iterator):
        sim_matrix[row_col[0], row_col[1]] = results[i]

    sim_matrix = sim_matrix + sim_matrix.T

    # Run multidimensional scaling to create a 2-Dimensional distance pane
    MDS_method = MDS(n_components=2, metric=True, n_init=4, max_iter=300, verbose=0, eps=0.001, n_jobs=-1,
                     random_state=None, dissimilarity='precomputed')
    X_MDS = MDS_method.fit_transform(sim_matrix)

    return X_MDS


def _flatten(input):
    """ Return """
    if isinstance(input, collections.Iterable):
        output = []
        for subitem in input:
            output += _flatten(subitem)
        return output
    else:
        return [input]


def _reorder_adjacency_matrix(X, r_list, plot_diag=True):
    new_order = _flatten(r_list)
    reordered_adj_mat = np.zeros_like(X)

    for r, i in enumerate(new_order):
        for c, j in enumerate(new_order):
            reordered_adj_mat[r, c] = X[i, j]

    if plot_diag:
        reordered_adj_mat = reordered_adj_mat + np.diag(np.ones(reordered_adj_mat.shape[0]))
    return reordered_adj_mat, new_order


def _assignment_array_to_lists(assignment_array):
    by_attribute_value = defaultdict(list)
    for node_index, attribute_value in enumerate(assignment_array):
        by_attribute_value[attribute_value].append(node_index)
    return list(by_attribute_value.values())


def _variation_of_information(a, b):
    a_entropy = mutual_info_score(a, a)
    b_entropy = mutual_info_score(b, b)
    mutual_information = mutual_info_score(a, b)
    return a_entropy + b_entropy - 2 * mutual_information
