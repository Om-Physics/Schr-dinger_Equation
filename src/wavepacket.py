#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
wavepacket.py
=============
Utilities for constructing quantum wavepackets commonly used in TDSE studies.

All quantities are in atomic units (ħ = mₑ = e = a₀ = 1).

Author : Om-Physics
License: MIT
"""

import numpy as np

__all__ = [
    "gaussian_wavepacket",
    "gaussian_wavepacket_2d",
    "plane_wave",
    "coherent_superposition",
]


def gaussian_wavepacket(x, x0=0.0, k0=0.0, sigma=0.1):
    """Normalised 1-D Gaussian wavepacket.

    The wavefunction is:

        ψ₀(x) = (π σ²)^{-1/4} · exp[i·k₀·(x−x₀)] · exp[−(x−x₀)²/(2σ²)]

    which satisfies ∫|ψ₀|² dx = 1 by construction.

    Parameters
    ----------
    x     : array_like — spatial grid (a.u.)
    x0    : float      — centre position of the wavepacket (a.u.)
    k0    : float      — average momentum / wave-vector (a.u.)
    sigma : float      — Gaussian width in position space (a.u.)

    Returns
    -------
    psi0 : complex ndarray, same shape as x
    """
    x   = np.asarray(x, dtype=float)
    env = np.sqrt(1.0 / (np.sqrt(np.pi) * sigma))
    env *= np.exp(-(x - x0)**2 / (2.0 * sigma**2))
    return np.exp(1j * k0 * (x - x0)) * env


def gaussian_wavepacket_2d(x, y, x0=0.0, y0=0.0, kx=0.0, ky=0.0,
                            sigma_x=0.1, sigma_y=0.1):
    """Normalised 2-D Gaussian wavepacket.

    ψ₀(x,y) = ψ_x(x) · ψ_y(y)  (product of two 1-D Gaussian wavepackets).

    Parameters
    ----------
    x, y     : 2-D array_like — meshgrid of spatial coordinates (a.u.)
    x0, y0   : float          — centre positions
    kx, ky   : float          — average momenta per axis
    sigma_x  : float          — width along x
    sigma_y  : float          — width along y

    Returns
    -------
    psi0 : complex ndarray, same shape as x
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    psi_x = gaussian_wavepacket(x, x0=x0, k0=kx, sigma=sigma_x)
    psi_y = gaussian_wavepacket(y, x0=y0, k0=ky, sigma=sigma_y)
    return psi_x * psi_y


def plane_wave(x, k0=1.0, L=1.0):
    """Box-normalised plane wave  ψ(x) = L^{-1/2} · exp(i·k₀·x).

    Parameters
    ----------
    x  : array_like — spatial grid
    k0 : float      — wave-vector
    L  : float      — domain length for normalisation

    Returns
    -------
    psi : complex ndarray
    """
    x = np.asarray(x, dtype=float)
    return np.exp(1j * k0 * x) / np.sqrt(L)


def coherent_superposition(x, weights, x0_list, k0_list, sigma_list):
    """Normalised coherent superposition of Gaussian wavepackets.

    ψ₀(x) = N · Σᵢ wᵢ · ψᵢ(x)   where N is a normalisation constant.

    Parameters
    ----------
    x        : array_like              — spatial grid
    weights  : list of float           — (possibly complex) mixing coefficients
    x0_list  : list of float           — centres
    k0_list  : list of float           — momenta
    sigma_list: list of float          — widths

    Returns
    -------
    psi0 : complex ndarray
    """
    x    = np.asarray(x, dtype=float)
    psi  = np.zeros_like(x, dtype=complex)
    for w, x0, k0, sigma in zip(weights, x0_list, k0_list, sigma_list):
        psi += w * gaussian_wavepacket(x, x0=x0, k0=k0, sigma=sigma)
    norm  = np.sqrt(np.trapezoid(np.abs(psi)**2, x))
    return psi / norm
