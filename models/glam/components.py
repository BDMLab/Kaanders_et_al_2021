#!/usr/bin/python

import theano.tensor as tt
import numpy as np


def tt_normal_cdf(x, mu=0, sd=1):
    """
    Normal cumulative distribution function
    Theano tensor implementation
    """
    return (0.5 + 0.5 * tt.erf((x - mu) / (sd * tt.sqrt(2.))))


def tt_wienerpos_fpt_pdf(t, drift, noise, boundary):
    """
    Probability density function of first passage times of
    Wiener process with positive drift towards constant boundary.
    Theano tensor implementation

    Cf https://en.wikipedia.org/wiki/Inverse_Gaussian_distribution#Relationship_with_Brownian_motion
    """
    mu = boundary / drift
    lam = (boundary**2 / noise**2)
    return ((lam / (2*np.pi*t**3))**0.5 * tt.exp((-lam * (t - mu)**2)/(2*mu**2*t)))


def tt_wienerpos_fpt_cdf(t, drift, noise, boundary, numerical_stability=100):
    """
    Cumulative distribution function of first passage times of
    Wiener process with positive drift towards constant boundary.
    Theano tensor implementation

    Cf https://en.wikipedia.org/wiki/Inverse_Gaussian_distribution#Relationship_with_Brownian_motion
    """
    mu = boundary / drift
    lam = (boundary / noise)**2
    bounded_ratio = tt.where(
        lam/mu >= numerical_stability, numerical_stability, lam/mu)
    return (tt_normal_cdf(tt.sqrt(lam / t) * (t / mu - 1)) +
            tt.exp(2*bounded_ratio) * tt_normal_cdf(-(tt.sqrt(lam / t) * (t / mu + 1))))


def tt_wienerrace_pdf(t, drift, noise, boundary, t0, zerotol=1e-14):
    """
    Probability density function of first passage times of
    a race between multiple Wiener processes with positive drift
    towards a constant boundary.
    Theano tensor implementation
    """
    # Nondecision time T0
    t = t - t0
    t = tt.where(t <= 1, 1, t)

    # first passage time densities, single Wiener accumulators
    f = tt_wienerpos_fpt_pdf(t, drift, noise, boundary)
    # first passage time distributions, single Wiener accumulators
    F = tt_wienerpos_fpt_cdf(t, drift, noise, boundary)
    # survival functions
    S = 1. - F
    # race densities
    # Note: drifts should be sorted so that chosen item drift is in first column
    l = f[:, 0] * tt.prod(S[:, 1:], axis=1)
    return l


def expdrift(v, tau, gamma, values, gaze, zerotol):
    """
    expdrift
    scaling between 0 and 10

    vectorized, i.e., runs on all trials simultaneously

    R = v * 10 / (1 + exp(-tau * (E_i - max(E_j))))
    """

    E_drifts = gaze * values + (1. - gaze) * gamma * values
    n_items = tt.cast(E_drifts.shape[1], dtype='int32')
    stacked_E = tt.repeat(E_drifts, repeats=n_items, axis=1).T
    stacked_E_reshaped = tt.reshape(stacked_E, newshape=(
        E_drifts.shape[1], E_drifts.shape[1], E_drifts.shape[0])).T
    identity = 1 - tt.eye(n_items)
    max_others = tt.max(stacked_E_reshaped * identity[None, :, :], axis=2)

    drift = E_drifts - max_others
    drift = v * 10 / (1 + tt.exp(-tau*drift))

    return drift
