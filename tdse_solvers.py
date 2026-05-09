#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tdse_solvers.py
===============
Numerical solvers for the 1D Time-Dependent Schrödinger Equation (TDSE).

Five methods are implemented, each following the derivation by Qijing Zheng
(USTC, 2022):

    1. Crank–Nicolson (CN)        — unitary, implicit, tridiagonal solve
    2. 4th-order Runge–Kutta (RK4)— explicit, norm-non-conserving for large dt
    3. Chebyshev polynomial        — spectral propagator, very accurate
    4. Lanczos / Krylov            — adaptive basis; large dt possible
    5. Split-Operator (SO)         — FFT-based, unconditionally unitary

All methods work in atomic units (ħ = mₑ = e = a₀ = 1).
Zero boundary conditions are enforced by CN/RK4/Chebyshev/Lanczos;
periodic boundary conditions are implicit in Split-Operator (FFT).

References
----------
[1] Crank–Nicolson method: https://en.wikipedia.org/wiki/Crank-Nicolson_method
[2] Runge–Kutta:           https://en.wikipedia.org/wiki/Runge-Kutta_methods
[3] Chebyshev polynomials: https://en.wikipedia.org/wiki/Chebyshev_polynomials
[4] Krylov / Lanczos:      J. Chem. Phys. 128, 164116 (2008)
[5] Split-operator:        Suzuki–Trotter expansion

Author : Om-Physics (adapted from Qijing Zheng, USTC)
License: MIT
"""

import numpy as np
import scipy.sparse as spa
from scipy.sparse.linalg import splu
from scipy.special import j0, j1, jv
from scipy.linalg import eigh
from scipy.fft import fft, ifft, fftfreq

__all__ = [
    "CrankNicolson",
    "RungeKutta",
    "ChebyshevPoly",
    "Lanczos",
    "SplitOperatorKin",
]


# ---------------------------------------------------------------------------
# Helper: build kinetic + potential operators on a 1-D grid
# ---------------------------------------------------------------------------

def _build_hamiltonian_1d(V_arr, dx):
    """Return sparse Hamiltonian H = T + V for a 1-D grid.

    Parameters
    ----------
    V_arr : 1-D array of length J+1 — external potential on the grid
    dx    : float — grid spacing (Δx)

    Returns
    -------
    H : (J+1)×(J+1) sparse matrix (CSR)
    T : kinetic part (CSR)
    V : potential part (diagonal, CSR)
    """
    J1 = len(V_arr)
    O  = np.ones(J1)
    T  = (-1.0 / (2.0 * dx**2)) * spa.spdiags(
        [O, -2*O, O], [-1, 0, 1], J1, J1, format='csr')
    V  = spa.diags(V_arr, format='csr')
    H  = T + V
    return H, T, V


# ===========================================================================
# 1.  Crank–Nicolson Method
# ===========================================================================

def CrankNicolson(psi0, V, x, dt, N=200, print_norm=False):
    """Propagate ψ using the Crank–Nicolson (Cayley) method.

    The unitary propagator is approximated as the Cayley form:

        U(Δt) ≈ (I − i·H·Δt/2) / (I + i·H·Δt/2)

    which is solved as a linear system at each time step:

        U₂ · Ψⁿ⁺¹ = U₁ · Ψⁿ
        U₂ = I + i·Δt/2·H
        U₁ = I − i·Δt/2·H

    Boundary condition: zero (infinite-well walls).

    Parameters
    ----------
    psi0 : array_like, shape (J+1,)  — initial wavefunction
    V    : array_like, shape (J+1,)  — external potential
    x    : array_like, shape (J+1,)  — spatial grid
    dt   : float                      — time step (a.u.)
    N    : int                        — number of time steps
    print_norm : bool                 — print norm at each step if True

    Returns
    -------
    PSI_t : complex array, shape (J+1, N) — ψ(x, t) for all time steps
    """
    x   = np.asarray(x, dtype=float)
    V   = np.asarray(V, dtype=float)
    J   = x.size - 1
    dx  = x[1] - x[0]

    H, T, Vop = _build_hamiltonian_1d(V, dx)

    U2 = (spa.eye(J+1, format='csc') + (1j * 0.5 * dt) * H).tocsc()
    U1 = (spa.eye(J+1, format='csc') - (1j * 0.5 * dt) * H).tocsc()
    LU = splu(U2)

    PSI_t = np.zeros((J+1, N), dtype=complex)
    PSI_t[:, 0] = np.asarray(psi0, dtype=complex)

    for n in range(N - 1):
        b              = U1.dot(PSI_t[:, n])
        PSI_t[:, n+1]  = LU.solve(b)
        if print_norm:
            norm = np.trapezoid(np.abs(PSI_t[:, n+1])**2, x)
            print(f"  step {n+1:4d}  norm = {norm:.8f}")

    return PSI_t


# ===========================================================================
# 2.  4th-order Runge–Kutta Method
# ===========================================================================

def RungeKutta(psi0, V, x, dt, N=200, print_norm=False):
    """Propagate ψ using the classical 4th-order Runge–Kutta (RK4) method.

    The TDSE is cast as  dψ/dt = f(t, ψ) = −i·H·ψ  and the standard RK4
    update formula is applied at each step.

    Note: RK4 does *not* conserve the norm exactly; a sufficiently small dt
    must be chosen to avoid divergence.  Recommended:  dt ≲ 1/E_max.

    Parameters
    ----------
    psi0 : array_like, shape (J+1,)
    V    : array_like, shape (J+1,)
    x    : array_like, shape (J+1,)
    dt   : float
    N    : int
    print_norm : bool

    Returns
    -------
    PSI_t : complex array, shape (J+1, N)
    """
    x   = np.asarray(x, dtype=float)
    V   = np.asarray(V, dtype=float)
    J   = x.size - 1
    dx  = x[1] - x[0]

    H, _, _ = _build_hamiltonian_1d(V, dx)
    U = -1j * H   # right-hand side operator: dψ/dt = U·ψ

    PSI_t = np.zeros((J+1, N), dtype=complex)
    PSI_t[:, 0] = np.asarray(psi0, dtype=complex)

    for n in range(N - 1):
        pn = PSI_t[:, n]
        k1 = dt * U.dot(pn)
        k2 = dt * U.dot(pn + 0.5 * k1)
        k3 = dt * U.dot(pn + 0.5 * k2)
        k4 = dt * U.dot(pn + k3)
        PSI_t[:, n+1] = pn + (k1 + 2*k2 + 2*k3 + k4) / 6.0
        if print_norm:
            norm = np.trapezoid(np.abs(PSI_t[:, n+1])**2, x)
            print(f"  step {n+1:4d}  norm = {norm:.8f}")

    return PSI_t


# ===========================================================================
# 3.  Chebyshev Polynomial Expansion
# ===========================================================================

def ChebyshevPoly(psi0, V, x, dt, N=200, delta=1e-10, print_norm=False):
    """Propagate ψ using the Chebyshev polynomial expansion of the propagator.

    The time-evolution operator is expanded as

        exp(−i·H·Δt) = J₀(τ)·I + 2 Σₙ (−i)ⁿ Jₙ(τ) Tₙ(H̃)

    where Jₙ are Bessel functions of the first kind, Tₙ are Chebyshev
    polynomials, τ = ‖H‖·Δt is the dimensionless time, and H̃ = H/‖H‖.
    The series is truncated once |Jₙ(τ)| < delta.

    Parameters
    ----------
    psi0  : array_like, shape (J+1,)
    V     : array_like, shape (J+1,)
    x     : array_like, shape (J+1,)
    dt    : float
    N     : int
    delta : float  — convergence threshold for truncating the series
    print_norm : bool

    Returns
    -------
    PSI_t : complex array, shape (J+1, N)
    """
    x   = np.asarray(x, dtype=float)
    V   = np.asarray(V, dtype=float)
    J   = x.size - 1
    dx  = x[1] - x[0]

    Hmax = float(V.max() + 1.0 / dx**2)   # spectral radius estimate
    tau  = Hmax * dt

    # Build Bessel coefficients until convergence
    Jn_tau = [j0(tau), -2j * j1(tau)]
    ii = 2
    while True:
        tmp = 2.0 * (-1j)**ii * jv(ii, tau)
        if np.abs(tmp) < delta:
            break
        Jn_tau.append(tmp)
        ii += 1
    Npoly = len(Jn_tau)
    print(f"[Chebyshev] Expansion truncated at N = {Npoly} terms.")

    H, _, _ = _build_hamiltonian_1d(V, dx)
    Hnorm   = H / Hmax   # normalised Hamiltonian ∈ [−1, 1]

    PSI_t        = np.zeros((J+1, N), dtype=complex)
    PSI_ShebyExp = np.zeros((J+1, Npoly), dtype=complex)
    PSI_t[:, 0]  = np.asarray(psi0, dtype=complex)

    for n in range(N - 1):
        PSI_ShebyExp[:, 0] = PSI_t[:, n]
        PSI_ShebyExp[:, 1] = Hnorm.dot(PSI_ShebyExp[:, 0])
        for jj in range(2, Npoly):
            PSI_ShebyExp[:, jj] = (2.0 * Hnorm.dot(PSI_ShebyExp[:, jj-1])
                                   - PSI_ShebyExp[:, jj-2])
        PSI_t[:, n+1] = PSI_ShebyExp.dot(Jn_tau)
        if print_norm:
            norm = np.trapezoid(np.abs(PSI_t[:, n+1])**2, x)
            print(f"  step {n+1:4d}  norm = {norm:.8f}")

    return PSI_t


# ===========================================================================
# 4.  Lanczos / Krylov Subspace Method
# ===========================================================================

def Lanczos(psi0, V, x, dt, N=200, M=15, print_norm=False):
    """Propagate ψ using the Lanczos method in the Krylov subspace.

    At each step, an M-dimensional Krylov subspace is built by Gram-Schmidt
    orthogonalisation.  The Hamiltonian is diagonalised in this reduced basis
    and the propagator is applied exactly there before mapping back to real
    space.  Larger M allows larger dt.

    Parameters
    ----------
    psi0 : array_like, shape (J+1,)
    V    : array_like, shape (J+1,)
    x    : array_like, shape (J+1,)
    dt   : float
    N    : int
    M    : int   — Krylov subspace dimension
    print_norm : bool

    Returns
    -------
    PSI_t : complex array, shape (J+1, N)
    """
    x   = np.asarray(x, dtype=float)
    V   = np.asarray(V, dtype=float)
    J   = x.size - 1
    dx  = x[1] - x[0]

    H, _, _ = _build_hamiltonian_1d(V, dx)

    PSI_t = np.zeros((J+1, N), dtype=complex)
    PSI_t[:, 0] = np.asarray(psi0, dtype=complex)

    # Lanczos basis and reduced Hamiltonian
    Vlanc = np.zeros((J+1, M), dtype=complex)
    Hlanc = np.zeros((M, M),   dtype=complex)
    e1    = np.r_[1.0, np.zeros(M - 1)]   # first standard basis vector

    for n in range(N - 1):
        norm0 = np.sqrt(np.trapezoid(np.abs(PSI_t[:, n])**2, x))
        Vlanc[:, 0] = PSI_t[:, n] / norm0
        Hlanc[0, 0] = np.trapezoid(Vlanc[:, 0].conj() * H.dot(Vlanc[:, 0]), x)

        for k in range(M - 1):
            Vlanc[:, k+1] = H.dot(Vlanc[:, k])
            for j in range(k + 1):
                if k - j <= 1:
                    Vlanc[:, k+1] -= Hlanc[j, k] * Vlanc[:, j]
            norm_k1 = np.sqrt(np.trapezoid(np.abs(Vlanc[:, k+1])**2, x))
            Vlanc[:, k+1] /= norm_k1

            Hlanc[k+1, k+1] = np.trapezoid(
                Vlanc[:, k+1].conj() * H.dot(Vlanc[:, k+1]), x)
            Hlanc[k+1, k]   = np.trapezoid(
                Vlanc[:, k+1].conj() * H.dot(Vlanc[:, k]), x)
            Hlanc[k, k+1]   = Hlanc[k+1, k].conj()

        # Diagonalise the reduced Hamiltonian
        EL, UL = eigh(Hlanc)

        # Propagate in Krylov subspace and map back
        phi_lanc = UL @ (np.exp(-1j * dt * EL) * (UL.conj().T @ e1))
        PSI_t[:, n+1] = Vlanc @ phi_lanc

        if print_norm:
            norm = np.trapezoid(np.abs(PSI_t[:, n+1])**2, x)
            print(f"  step {n+1:4d}  norm = {norm:.8f}")

    return PSI_t


# ===========================================================================
# 5.  Split-Operator Method (kinetic-referenced, 2nd-order Suzuki–Trotter)
# ===========================================================================

def SplitOperatorKin(psi0, V, L, dt, N=200):
    """Propagate ψ using the kinetic-referenced split-operator technique.

    Each time step applies three sub-operations (Suzuki–Trotter, 2nd order):

        Ψⁿ⁺¹ = exp(−i·V·Δt/2) · IFFT[ exp(−i·k²/2·Δt) · FFT[ exp(−i·V·Δt/2)·Ψⁿ ] ]

    The potential operator is diagonal in real space; the kinetic operator is
    diagonal in reciprocal space — no matrix inversion needed.

    Boundary condition: **periodic** (implicit from FFT).  This differs from
    the zero-BC of CN/RK4/Chebyshev/Lanczos and manifests when the wavepacket
    reaches the domain boundary.

    Parameters
    ----------
    psi0 : array_like, shape matching V  — initial wavefunction
    V    : array_like, 1-D or 2-D        — external potential
    L    : list of floats                 — domain lengths per dimension, e.g. [L] or [Lx, Ly]
    dt   : float                          — time step (a.u.)
    N    : int                            — number of time steps

    Returns
    -------
    PSI_t : complex array, shape V.shape + (N,)
    """
    psi0 = np.asarray(psi0, dtype=complex)
    V    = np.asarray(V,    dtype=float)
    assert psi0.shape == V.shape
    assert len(L) == V.ndim

    # Build reciprocal-space grid (dimensionless integer frequencies)
    ff = [fftfreq(n, 1.0/n) for n in V.shape]

    if V.ndim == 1:
        k2 = (2.0 * np.pi / L[0])**2 * ff[0]**2
    elif V.ndim == 2:
        kx, ky = np.meshgrid(ff[0], ff[1], indexing='ij')
        k2 = ((2.0*np.pi/L[0])**2 * kx**2 +
              (2.0*np.pi/L[1])**2 * ky**2)
    else:
        raise NotImplementedError("SplitOperatorKin supports 1-D and 2-D only.")

    Ut_full = np.exp(-1j * dt * k2 / 2.0)   # kinetic propagator (full step)
    Uv_half = np.exp(-1j * dt * V  / 2.0)   # potential propagator (half step)

    PSI_t          = np.zeros(psi0.shape + (N,), dtype=complex)
    PSI_t[..., 0]  = psi0

    for n in range(N - 1):
        psi_half        = Uv_half * PSI_t[..., n]
        psi_k           = fft(psi_half)
        psi_k_evolved   = Ut_full * psi_k
        psi_real        = ifft(psi_k_evolved)
        PSI_t[..., n+1] = Uv_half * psi_real

    return PSI_t


# ===========================================================================
# Utility: suggest a safe time step for a given wavepacket
# ===========================================================================

def estimate_dt(k0, sigma_x):
    """Estimate a safe time step from the wavepacket parameters.

    Based on the kinetic energy of the highest-momentum plane wave in the
    Gaussian wavepacket (k₀ ± σ_k with σ_k = 1/σ_x in a.u.):

        Δt ≈ 1 / E_k = 2 / (k₀ + σ_k)²

    Parameters
    ----------
    k0      : float — average momentum (a.u.)
    sigma_x : float — wavepacket width in position space (a.u.)

    Returns
    -------
    dt : float — recommended time step
    """
    sigma_k = 1.0 / sigma_x
    E_k = 0.5 * (abs(k0) + sigma_k)**2
    return 1.0 / E_k if E_k > 0 else 0.01
