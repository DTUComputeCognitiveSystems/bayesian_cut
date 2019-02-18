import numpy as np
import glob, os
from scipy.io import loadmat
from scipy.sparse.csgraph import connected_components
import scipy.sparse as sp
import networkx as nx


def load_data(network='karate', labels=False, remove_disconnected=False, get_gml=False):
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

    try:
        filetype = glob.glob(os.path.join(ROOT_PATH, network) + '.*')[0].split('.')[1]
    except:
        print("The requested network could not be found")
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
