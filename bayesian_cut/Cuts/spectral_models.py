#!/usr/bin/env python3
#
# -*- coding: utf-8 -*-
#
# Author:   Laurent Vermue <lauve@dtu.dk>
#           Maciej Korzepa <mjko@dtu.dk>
#           Petr Taborsky <ptab@dtu.dk>
#
# License: 3-clause BSD

import scipy as sc
from sklearn.cluster import KMeans
from numpy.linalg import eigh
import numpy as np

class base_class(object):
    def __init__(self, X):
        self.X = X.toarray()
        self.z_ = None


class RatioCut(base_class):
    """
    Cuts a graph with the RatioCut method using its adjacency matrix X of size nxn

    Model settings
    --------------

    X : sparse scipy matrix, shape(n, n)
        Adjacency matrix

    Model attributes after running
    ------------------------------

    z_ : numpy array, shape(n,)
        Group assignment vector

    Reference:
    Von Luxburg, Ulrike. "A tutorial on spectral clustering." Statistics and computing 17.4 (2007): 395-416.
    """
    def run(self, add_noiselinks=False):
        """ Perform given cut method on the adjacency matrix X

        Parameters
        ----------

        add_noiselinks : float
            Percentage of all links to be altered. Example: The adjacency matrix contains 100 links and add_noiselinks
            is set to 0.1. In this case 10 links are altered, i.e. existing links can disappear or new links appear.
        """
        X = self.X.copy()
        if add_noiselinks > 0:
            add_noiselinks = int(add_noiselinks * X.sum() / 2)
            indices = np.triu_indices(X.shape[0], 1)
            choices = np.random.choice(indices[0].shape[0], add_noiselinks, replace=False)
            indices = np.array(indices)
            indices = indices[:, choices]
            indices = (indices[0], indices[1])
            X[indices] = 1 - X[indices]
            lower_indices = np.tril_indices(X.shape[0], 0)
            X[lower_indices] = 0
            X = X + X.T

        D = np.diagflat(np.sum(X, axis=0))  # degree matrix nxn
        L = D - X  # unnormalized Laplacian
        k = 2  # simplified problem with k set to 2
        (eigvals, eigvects) = np.linalg.eigh(L)  # eigenvectors and coresponding eigenvalues of L
        U = eigvects[:, eigvals.argsort()][:, 0:k]
        RR = KMeans(n_clusters=k).fit(U)  # clustering rows of U (nxk matrix) having eigenvectors in columns
        self.z_ = RR.predict(U)


class NormCutSM(base_class):
    """
    Cuts a graph with the NormCut method by Shi and Malik method using its adjacency matrix X of size nxn

    Model settings
    --------------

    X : sparse scipy matrix, shape(n, n)
        Adjacency matrix

    Model attributes after running
    ------------------------------

    z_ : numpy array, shape(n,)
        Group assignment vector

    Reference:
    Von Luxburg, Ulrike. "A tutorial on spectral clustering." Statistics and computing 17.4 (2007): 395-416.
    """
    def run(self, add_noiselinks=False):
        """ Perform given cut method on the adjacency matrix X

        Parameters
        ----------

        add_noiselinks : float
            Percentage of all links to be altered. Example: The adjacency matrix contains 100 links and add_noiselinks
            is set to 0.1. In this case 10 links are altered, i.e. existing links can disappear or new links appear.
        """
        X = self.X.copy()
        if add_noiselinks > 0:
            add_noiselinks = int(add_noiselinks * X.sum() / 2)
            indices = np.triu_indices(X.shape[0], 1)
            choices = np.random.choice(indices[0].shape[0], add_noiselinks, replace=False)
            indices = np.array(indices)
            indices = indices[:, choices]
            indices = (indices[0], indices[1])
            X[indices] = 1 - X[indices]
            lower_indices = np.tril_indices(X.shape[0], 0)
            X[lower_indices] = 0
            X = X + X.T

        D = np.diagflat(np.sum(X, axis=1))  # degree matrix nxn
        L = D - X  # unnormalized Laplacian

        k = 2  # simplified problem with k set to 2
        (eigvals, eigvects) = sc.linalg.eigh(a=L, b=D)
        # eigenvectors and coresponding eigenvalues of generalized eigenproblem Lu=lambdaDu
        U = eigvects[:, eigvals.argsort()][:, 0:k]
        RR = KMeans(n_clusters=k).fit(U)  # clustering rows of U (nxk matrix) having eigenvectors in columns
        self.z_ = RR.predict(U)


class NormCutNJW(base_class):
    """
    Cuts a graph with the NormCut method by NG, Jordan and Weiss using its adjacency matrix X of size nxn

    Model settings
    --------------

    X : sparse scipy matrix, shape(n, n)
        Adjacency matrix

    Model attributes after running
    ------------------------------

    z_ : numpy array, shape(n,)
        Group assignment vector

    Reference:
    Von Luxburg, Ulrike. "A tutorial on spectral clustering." Statistics and computing 17.4 (2007): 395-416.
    """
    def run(self, add_noiselinks=False):
        """ Perform given cut method on the adjacency matrix X

        Parameters
        ----------

        add_noiselinks : float
            Percentage of all links to be altered. Example: The adjacency matrix contains 100 links and add_noiselinks
            is set to 0.1. In this case 10 links are altered, i.e. existing links can disappear or new links appear.
        """
        X = self.X.copy()
        if add_noiselinks > 0:
            add_noiselinks = int(add_noiselinks * X.sum() / 2)
            indices = np.triu_indices(X.shape[0], 1)
            choices = np.random.choice(indices[0].shape[0], add_noiselinks, replace=False)
            indices = np.array(indices)
            indices = indices[:, choices]
            indices = (indices[0], indices[1])
            X[indices] = 1 - X[indices]
            lower_indices = np.tril_indices(X.shape[0], 0)
            X[lower_indices] = 0
            X = X + X.T
        D = np.diagflat(np.sum(X, axis=1))  # degree matrix nxn
        D1_2 = np.linalg.cholesky(np.linalg.inv(D))
        L = np.matlib.eye(X.shape[0]) - np.dot(np.dot(D1_2, X), D1_2)  # symetric Laplacian

        k = 2  # simplified problem with k set to 2
        (eigvals, eigvects) = sc.linalg.eigh(L)  # eigenvectors and coresponding eigenvalues
        U = eigvects[:, eigvals.argsort()][:, 0:k]
        Uu = np.matrix(np.sqrt(np.sum(np.power(U, 2), axis=1)))
        T = np.multiply(U, (1 / Uu).T)
        # check of the norm equals one: np.sqrt(np.sum(np.power(T,2),axis=1))
        RR = KMeans(n_clusters=k).fit(T)  # clustering rows of U (nxk matrix) having eigenvectors in columns
        self.z_ = RR.predict(T)


class NewmanModularityCut(base_class):
    """
    Cuts a graph with the spectral modularity optimization method by Newman using its adjacency matrix X of size nxn

    Model settings
    --------------

    X : sparse scipy matrix, shape(n, n)
        Adjacency matrix

    Model attributes after running
    ------------------------------

    z_ : numpy array, shape(n,)
        Group assignment vector

    Reference:
    Newman, M. E. J. “Modularity and Community Structure in Networks.” Proceedings of the National Academy of
    Sciences of the United States of America 103.23 (2006): 8577–8582. PMC. Web. 2 Oct. 2018.
    """
    def run(self, add_noiselinks=False):
        """ Perform given cut method on the adjacency matrix X

                Parameters
                ----------

                add_noiselinks : float
                    Percentage of all links to be altered. Example: The adjacency matrix contains 100 links and add_noiselinks
                    is set to 0.1. In this case 10 links are altered, i.e. existing links can disappear or new links appear.
                """
        A = self.X.copy()
        np.fill_diagonal(A, 0)
        if add_noiselinks > 0:
            add_noiselinks = int(add_noiselinks * A.sum() / 2)
            indices = np.triu_indices(A.shape[0], 1)
            choices = np.random.choice(indices[0].shape[0], add_noiselinks, replace=False)
            indices = np.array(indices)
            indices = indices[:, choices]
            indices = (indices[0], indices[1])
            A[indices] = 1 - A[indices]
            lower_indices = np.tril_indices(A.shape[0], 0)
            A[lower_indices] = 0
            A = A + A.T

        degree_vec = np.sum(A, axis=1)
        m = np.sum(degree_vec) / 2
        deduct_matrix = np.outer(degree_vec, degree_vec) / (2 * m)
        B = A - deduct_matrix
        w, v = eigh(B)
        leading_eig_value = np.argmax(w)
        z_ = np.squeeze(np.array(v[:, leading_eig_value]))
        self.z_ = np.zeros((A.shape[0]))
        self.z_[z_ >= 0] = 1
        self.z_[z_ < 0] = -1
        modularity = (1/(4*m)) * self.z_.T.dot(B).dot(self.z_)
        print("Modularity of method Newman Cut: {:.4f}".format(modularity))
        self.z_[z_ < 0] = 0
