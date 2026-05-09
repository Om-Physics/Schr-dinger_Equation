"""
TDSE — Numerical Solutions to the Time-Dependent Schrödinger Equation
======================================================================
A self-contained Python package implementing five propagation schemes:

    • Crank–Nicolson (implicit, unitary, tridiagonal solve)
    • 4th-order Runge–Kutta (explicit, conditionally stable)
    • Chebyshev polynomial expansion (spectral propagator)
    • Lanczos / Krylov subspace method (adaptive basis)
    • Split-Operator technique (FFT-based, unconditionally unitary)

Companion utilities cover wavepacket construction and standard potentials.

All quantities are in atomic units (ħ = mₑ = e = a₀ = 1).
"""

from .tdse_solvers import (
    CrankNicolson,
    RungeKutta,
    ChebyshevPoly,
    Lanczos,
    SplitOperatorKin,
    estimate_dt,
)
from .wavepacket import (
    gaussian_wavepacket,
    gaussian_wavepacket_2d,
    plane_wave,
    coherent_superposition,
)
from .potentials import (
    zero_potential,
    finite_barrier,
    harmonic_oscillator,
    double_well,
    finite_square_well,
    step_potential,
    morse_potential,
    coulomb_soft,
    harmonic_oscillator_2d,
    double_slit_2d,
)

__version__ = "1.0.0"
__author__  = "Om-Physics"
