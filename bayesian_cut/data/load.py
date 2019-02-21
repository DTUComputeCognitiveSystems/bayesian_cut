#!/usr/bin/env python3
#
# -*- coding: utf-8 -*-
#
# Author:   Maciej Korzepa <mjko@dtu.dk>
#           Laurent Vermue <lauve@dtu.dk>
#           Petr Taborsky <ptab@dtu.dk>
#
# License: 3-clause BSD

import numpy as np
import glob, os
from scipy.io import loadmat
from scipy.sparse.csgraph import connected_components
import scipy.sparse as sp
import networkx as nx
import time
from urllib.request import urlretrieve
import sys

GITHUB_DATADIR = 'https://github.com/DTUComputeCognitiveSystems/bayesian_cut/raw/master/bayesian_cut/data'


def load_data(network='karate', labels=True, remove_disconnected=False, get_gml=False):
    """
    This function load the requested network as sparse scipy matrix and returns it together
    with the cluster labels, if these were requested
    :param network: The name of the network
    :param labels: Load labels for network (only if available)
    :param remove_disconnected: Bool: Whether disconnected components should be removed
    :param keep_gml: Bool: Whether loaded gml networks should also be returned as gml networks
    :return: (if keep_gml==False) Scipy sparse csr matrix of the network, Numpy array of the labels (None if not requested)
    :return: (if keep_gml==True) Adjusted gml network
    """
    # First load the adjacency matrix
    global cc_idx
    ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
    Y = None

    # try:
    #     filetype = glob.glob(os.path.join(ROOT_PATH, network) + '.*')[0].split('.')[1]
    # except:
    #     print("The requested network could not be found")
    #     return None, None
    
    filetype = 'mat' if network == 'karate' else 'gml'
    abs_file_path = os.path.join(ROOT_PATH, '{0}.{1}'.format(network, filetype))

    if file_checker('{0}.{1}'.format(network, filetype), abs_file_path, GITHUB_DATADIR) != 0:
        print('File {0}.{1} was not available and could not be downloaded'.format(network, filetype))
        return None, None

    if filetype == 'mat':
        X = loadmat(os.path.join(ROOT_PATH, network) + '.' + filetype)
        X = X['Problem'][0][0][2].tocsr()  # sparse matrix
    elif filetype == 'txt':
        data = np.loadtxt(os.path.join(ROOT_PATH, network) + '.' + filetype)
        data = np.concatenate((data, data[:, [1, 0]]))
        data = np.unique([tuple(row) for row in data], axis=0)
        N = int(np.max(data) - np.min(data) +1)
        row, col = data[:, 0], data[:, 1]
        data = np.ones((row.size,))
        X = sp.coo_matrix((data, (row, col)), shape=(N, N)).tocsr()
    elif filetype == 'gml':
        G = nx.OrderedGraph(nx.read_gml(os.path.join(ROOT_PATH, network) + '.' + filetype))
        print('Original graph size:', G.number_of_nodes())
        G = max(nx.connected_component_subgraphs(G), key=len)  # keep only largest connected component
        G.remove_edges_from(G.selfloop_edges())  # remove self-loops
        G = nx.convert_node_labels_to_integers(G, ordering='sorted')
        print('Largest cc:', G.number_of_nodes())
        if labels:
            nodes = sorted(G.nodes())
            # nodes = G.nodes()
            nodes_class = nx.get_node_attributes(G, 'value')
            Y = np.array([nodes_class[node] for node in nodes])
    else:
        print("This filetype has not been implemented yet.")
        return None

    if filetype != 'gml':
        # Force diagonal to zeros
        X.setdiag(0)

        if remove_disconnected:
            cc_idx = np.where(connected_components(X, False)[1] == 0)[0]
            X = X[cc_idx[:, None], cc_idx]

        if labels:
            # Second load the label vector
            try:
                filetype = glob.glob(os.path.join(ROOT_PATH, network) + '_labels' + '.*')[0].split('.')[1]
                if filetype == 'txt':
                    Y = np.loadtxt(os.path.join(ROOT_PATH, network) + '_labels.' + filetype)
                    # Remove the nodes from Y that were removed from X
                    if remove_disconnected:
                        Y = Y[cc_idx]
                else:
                    print("This filetype of the labels object has not been implemented yet.")
            except:
                print("Could not find a labels file for this network, None will be returned")


        assert np.allclose(X.data, X.T.data), "X not symmetrical!"
        assert np.sum(X.diagonal()) == 0, 'X diagonal = ' + str(X.diagonal())

    if get_gml:
        if filetype != 'gml':
            G = nx.from_scipy_sparse_matrix(X)
        return G, Y
    else:
        if filetype == 'gml':
            X = nx.to_scipy_sparse_matrix(G, nodelist=nodes)
        return X, Y

def file_checker(file, abs_file_path, github_dir):
    if os.path.isfile(abs_file_path):
        pass
    else: #Download the file from the github repository
        url = '{0}/{1}'.format(github_dir, file)
        print('File not available locally. Trying to download file {0} from github repository.\n Link: {1}'
              .format(file, url))
        try:
            urlretrieve(url, abs_file_path, reporthook)
        except:
            print("The file could not be downloaded. Please check your internet connection.")
            return 1
    return 0

# From https://blog.shichao.io/2012/10/04/progress_speed_indicator_for_urlretrieve_in_python.html
def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    # Adding small value for duration, to avoid division by zero
    speed = int(progress_size / (1024 * duration + 0.00001))
    percent = min(int(count*block_size*100/total_size), 100)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed\n" %
                    (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()
