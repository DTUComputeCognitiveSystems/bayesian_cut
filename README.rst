Bayesian Cut Package
====================

.. image:: https://img.shields.io/pypi/v/bayesian_cut.svg
    :target: https://pypi.python.org/pypi/mbpls
    :alt: Pypi Version
.. image:: https://img.shields.io/pypi/l/bayesian_cut.svg
    :target: https://pypi.python.org/pypi/mbpls/
    :alt: License

The Bayesian Cut Python package provides an easy to use API for the straight-forward application of Bayesian network
cuts using a full Bayesian inference framework based on the Gibbs-Sampler using the degree corrected Stochastic
Blockmodel (dc-SBM) or the Bayesian Cut (BC).
Furthermore it provides modularity, ratio-cut and norm cut based spectral network cut methods.
It also provides a rich visualization library that allow an easy analysis of posterior solution landscapes and network
cuts obtained by the various methods.

Jupyter Notebooks with examples on how to use the package can be found at
https://github.com/DTUComputeCognitiveSystems/bayesian_cut/tree/master/examples


Installation
------------

-  | Install the package for Python3 using the following command. Some
     dependencies might require an upgrade (scikit-learn, numpy and
     scipy).
   | ``$ pip install bayesian_cut``

-  | Now you can import the bayesian cut class by typing
   | ``from bayesian_cut.cuts import Model``

Quick Start
-----------

Use the bayesian_cut package for performing graph cuts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   # Load the package
   from bayesian_cut.data.load import load_data
   from bayesian_cut.cuts.bayesian_models import Model

   # Load an example network
   network_name = 'karate'
   X, Y = load_data(network=network_name, labels=True, remove_disconnected=True)

   # Set the parameters for the model
   n_samples_per_chain = 75
   n_chains = 15
   C = 2
   model_params = {
       'alpha_in': 1e-2,
       'beta_in': 1e-2,
       'alpha_out': 1e-2,
       'beta_out': 1e-2,
       'b': 1,
       'gamma': 3
   }

   # Define the model
   BC = Model('ShiftedApproximateBayesianCommunityDetection',
               X,
               model_params,
               Y=Y,
               C=C,
               block_sampling=False,
               marginalize_phi=True
               )

   # Add the number of chains to run
   BC.add_chains(number_of_chains=n_chains)

   # Run the Gibbs sampler
   BC.run_chains(n_samples=n_samples_per_chain,
                 n_prior_updates=20,
                 verbose=2,
                 save_all_samples=False,
                 parallel=True
                 )

   # Obtain the cut with the highest log-likelihood
   graph_cut_z_vector = BC.get_best_chain().max_log_lik_z_

   # Plot the cut as an adjacency matrix
   from bayesian_cut.utils import utils
   utils.Cluster_plot(BC)

   # Done
