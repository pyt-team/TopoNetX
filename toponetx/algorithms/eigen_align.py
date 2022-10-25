# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 16:29:54 2022

refs:
https://github.com/danielegrattarola/GINR/blob/master/src/utils/eigenvectors.py
"""


import itertools

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy

__all__ = [
    "compute_hist",
    "compute_kl",
    "compute_js",
    "compute_alignment",
    "align_eigenvectors_kl",
]


def compute_hist(ref, test):
    r_min = min(ref.min(), test.min())
    r_max = max(ref.max(), test.max())
    params = dict(bins=100, density=True, range=(r_min, r_max))
    h_ref, _ = np.histogram(ref, **params)
    h_test, _ = np.histogram(test, **params)
    h_test[np.isnan(h_test)] = 0.0
    h_test[np.isinf(h_test)] = 1e8
    h_test = np.where(h_test == 0.0, 1e-8, h_test)

    return h_ref, h_test


def compute_kl(ref, test):
    h_ref, h_test = compute_hist(ref, test)
    return entropy(h_ref, h_test)


def compute_js(ref, test):
    h_ref, h_test = compute_hist(ref, test)
    return jensenshannon(h_ref, h_test)


def compute_alignment(u1, u2):
    k = u1.shape[1]
    alignment = np.zeros((k, k))
    for i, j in itertools.product(range(k), range(k)):
        d = compute_js(u1[:, i], u2[:, j])
        alignment[i, j] = 1e-8 if np.isnan(d) else d

    return alignment


def align_eigenvectors_kl(u_ref, u_test):
    switch = []
    # Take care of first eigv manually
    switch.append(np.sign(u_ref[0, 0]) != np.sign(u_test[0, 0]))

    # Take care of other eigv
    for k in range(1, u_ref.shape[-1]):
        ref = u_ref[:, k]
        test = u_test[:, k]

        entropy_test = compute_kl(ref, test)
        entropy_test_minus = compute_kl(ref, -test)

        switch.append(entropy_test > entropy_test_minus)

    u_test[:, switch] = -u_test[:, switch]
    return u_test
