#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
potentials.py
=============
Common 1-D and 2-D potential energy functions for TDSE studies.
All values returned are in atomic units.

Author : Om-Physics
License: MIT
"""

import numpy as np

__all__ = [
    "zero_potential",
    "finite_barrier",
    "harmonic_oscillator",
    "double_well",
    "finite_square_well",
    "step_potential",
    "morse_potential",
    "coulomb_soft",
    "harmonic_oscillator_2d",
    "double_slit_2d",
]


# ---------------------------------------------------------------------------
# 1-D Potentials
# ---------------------------------------------------------------------------

def zero_potential(x):
    """V(x) = 0  (free particle / infinite-well in zero-BC solvers)."""
    return np.zeros_like(x, dtype=float)


def finite_barrier(x, x_centre=0.0, width=2.0, height=10.0):
    """Rectangular barrier of given width and height centred at x_centre.

    Parameters
    ----------
    x        : array_like
    x_centre : float — barrier centre (a.u.)
    width    : float — barrier full width (a.u.)
    height   : float — barrier height V₀ (a.u.)

    Returns
    -------
    V : ndarray
    """
    x = np.asarray(x, dtype=float)
    V = np.zeros_like(x)
    mask = np.abs(x - x_centre) <= width / 2.0
    V[mask] = height
    return V


def harmonic_oscillator(x, omega=1.0, x0=0.0):
    """V(x) = ½·ω²·(x−x₀)²  — harmonic oscillator well.

    Parameters
    ----------
    x     : array_like
    omega : float — angular frequency (a.u.)
    x0    : float — equilibrium position

    Returns
    -------
    V : ndarray
    """
    x = np.asarray(x, dtype=float)
    return 0.5 * omega**2 * (x - x0)**2


def double_well(x, a=1.0, b=0.5):
    """V(x) = −a·x² + b·x⁴  — symmetric double-well potential.

    Parameters
    ----------
    x : array_like
    a : float — negative quadratic coefficient
    b : float — positive quartic coefficient

    Returns
    -------
    V : ndarray (shifted so minimum = 0)
    """
    x = np.asarray(x, dtype=float)
    V = -a * x**2 + b * x**4
    return V - V.min()


def finite_square_well(x, x_centre=0.0, width=2.0, depth=10.0):
    """Finite square well (attractive box) of given depth.

    Parameters
    ----------
    x        : array_like
    x_centre : float — well centre
    width    : float — well full width (a.u.)
    depth    : float — well depth V₀ > 0  (a.u.)

    Returns
    -------
    V : ndarray  (V = 0 outside, −V₀ inside)
    """
    x = np.asarray(x, dtype=float)
    V = np.zeros_like(x)
    mask = np.abs(x - x_centre) <= width / 2.0
    V[mask] = -depth
    return V


def step_potential(x, x_step=0.0, height=5.0):
    """Heaviside step potential.

        V(x) = 0 for x < x_step,  V₀ for x ≥ x_step.

    Parameters
    ----------
    x      : array_like
    x_step : float — position of the step
    height : float — potential height V₀ (a.u.)
    """
    x = np.asarray(x, dtype=float)
    return np.where(x >= x_step, height, 0.0)


def morse_potential(x, De=10.0, a=1.0, x0=0.0):
    """Morse potential  V(x) = Dₑ·(1 − e^{−a(x−x₀)})².

    Parameters
    ----------
    x   : array_like
    De  : float — dissociation energy (a.u.)
    a   : float — controls well curvature
    x0  : float — equilibrium position

    Returns
    -------
    V : ndarray
    """
    x = np.asarray(x, dtype=float)
    return De * (1.0 - np.exp(-a * (x - x0)))**2


def coulomb_soft(x, Z=1.0, eps=1.0):
    """Soft-core Coulomb potential  V(x) = −Z / √(x² + ε²).

    Avoids the singularity at x = 0 via the softening parameter ε.

    Parameters
    ----------
    x   : array_like
    Z   : float — nuclear charge
    eps : float — softening (a.u.)

    Returns
    -------
    V : ndarray
    """
    x = np.asarray(x, dtype=float)
    return -Z / np.sqrt(x**2 + eps**2)


# ---------------------------------------------------------------------------
# 2-D Potentials
# ---------------------------------------------------------------------------

def harmonic_oscillator_2d(x, y, omega_x=1.0, omega_y=1.0, x0=0.0, y0=0.0):
    """2-D harmonic oscillator  V(x,y) = ½ω_x²(x−x₀)² + ½ω_y²(y−y₀)²."""
    return (0.5 * omega_x**2 * (x - x0)**2 +
            0.5 * omega_y**2 * (y - y0)**2)


def double_slit_2d(x, y, x_wall=0.0, wall_thickness=0.1,
                   slit_half_width=0.3, slit_separation=1.2,
                   height=1000.0):
    """2-D double-slit barrier.

    Parameters
    ----------
    x, y             : 2-D array_like — coordinate meshgrids
    x_wall           : float — x-position of the wall
    wall_thickness   : float — wall thickness in x
    slit_half_width  : float — half-width of each slit opening in y
    slit_separation  : float — centre-to-centre separation of the slits
    height           : float — barrier height (a.u.)

    Returns
    -------
    V : ndarray, same shape as x
    """
    V = np.zeros_like(x, dtype=float)
    in_wall = np.abs(x - x_wall) < wall_thickness / 2.0
    y_c1 =  slit_separation / 2.0
    y_c2 = -slit_separation / 2.0
    in_slit1 = np.abs(y - y_c1) < slit_half_width
    in_slit2 = np.abs(y - y_c2) < slit_half_width
    V[in_wall & ~in_slit1 & ~in_slit2] = height
    return V
