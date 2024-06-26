# -*- coding: utf-8 -*-
"""EM_bernoulli.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1VREHnpzVA79KZAuoKwZADk7JkAW6R4zu

# **EM on Bernoulli Mixture**
"""

import numpy as np
from scipy.stats import bernoulli

def initialize_parameters(k, n_features):
    """
    Initialize parameters for the EM algorithm.

    Parameters:
    - k: number of clusters
    - n_features: number of features

    Returns:
    - pi: initial mixture coefficients
    - theta: initial Bernoulli parameters for each cluster
    """
    pi = np.ones(k) / k  # Initialize uniform mixture coefficients
    theta = np.random.rand(k, n_features)  # Initialize random Bernoulli parameters
    return pi, theta

def expectation_step(X, pi, theta):
    """
    Perform the expectation step of the EM algorithm.

    Parameters:
    - X: input data (n_samples x n_features)
    - pi: mixture coefficients (k,)
    - theta: Bernoulli parameters for each cluster (k x n_features)

    Returns:
    - gamma: posterior probabilities (n_samples x k)
    """
    n_samples, n_features = X.shape
    k = len(pi)
    gamma = np.zeros((n_samples, k))

    for i in range(n_samples):
        for j in range(k):
            likelihood = np.prod(theta[j] ** X[i] * (1 - theta[j]) ** (1 - X[i]))
            gamma[i, j] = pi[j] * likelihood

        gamma[i] /= np.sum(gamma[i])  # Normalize to get posterior probabilities

    return gamma

def maximization_step(X, gamma):
    """
    Perform the maximization step of the EM algorithm.

    Parameters:
    - X: input data (n_samples x n_features)
    - gamma: posterior probabilities (n_samples x k)

    Returns:
    - pi_new: updated mixture coefficients (k,)
    - theta_new: updated Bernoulli parameters for each cluster (k x n_features)
    """
    n_samples, n_features = X.shape
    k = gamma.shape[1]

    Nk = np.sum(gamma, axis=0)
    pi_new = Nk / n_samples

    theta_new = np.zeros((k, n_features))
    for j in range(k):
        theta_new[j] = np.sum(gamma[:, j][:, np.newaxis] * X, axis=0) / Nk[j]

    return pi_new, theta_new

def em_algorithm(X, k, max_iter=100, tol=1e-4):
    """
    Perform the Expectation-Maximization (EM) algorithm for a mixture of Bernoulli distributions.

    Parameters:
    - X: input data (n_samples x n_features)
    - k: number of clusters
    - max_iter: maximum number of iterations
    - tol: tolerance for convergence

    Returns:
    - pi: final mixture coefficients (k,)
    - theta: final Bernoulli parameters for each cluster (k x n_features)
    """
    n_samples, n_features = X.shape
    pi, theta = initialize_parameters(k, n_features)

    for _ in range(max_iter):
        gamma = expectation_step(X, pi, theta)
        pi_new, theta_new = maximization_step(X, gamma)

        if np.linalg.norm(pi_new - pi) < tol and np.all(np.abs(theta_new - theta) < tol):
            break

        pi, theta = pi_new, theta_new

    return pi, theta

# Example usage
np.random.seed(0)
n_samples = 1000
n_features = 3
k = 2

# Generate synthetic data
X = np.random.randint(0, 2, size=(n_samples, n_features))

# Run EM algorithm
pi, theta = em_algorithm(X, k)
print("Final mixture coefficients:", pi)
print("Final Bernoulli parameters:")
print(theta)