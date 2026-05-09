# Numerical Solutions to the Time-Dependent Schrödinger Equation

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![NumPy](https://img.shields.io/badge/NumPy-1.23%2B-blue)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.9%2B-blue)](https://scipy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.5%2B-blue)](https://matplotlib.org/)

A complete, well-documented Python implementation of five numerical methods for propagating quantum wavefunctions governed by the Time-Dependent Schrödinger Equation (TDSE). All computations are performed in **atomic units** ($\hbar = m_e = e = a_0 = 1$).

---

## Table of Contents

- [Overview](#overview)
- [Methods Implemented](#methods-implemented)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Scripts and Figures](#scripts-and-figures)
- [Theory Summary](#theory-summary)
- [Results](#results)
- [References](#references)

---

## Overview

The TDSE describes the quantum-mechanical evolution of a particle under an external potential:

$$
i \frac{\partial}{\partial t} \psi(\mathbf{r}, t)
= \left[ -\frac{1}{2}\nabla^2 + V(\mathbf{r}) \right] \psi(\mathbf{r}, t)
$$

Its formal solution involves applying the unitary time-evolution operator $\mathcal{U}(\Delta t) = e^{-i\mathcal{H}\Delta t}$ to the wavefunction at each time step. This repository provides five distinct numerical strategies for approximating this operation, along with companion utilities for wavepacket construction, potential definition, figure generation, and animation.

---

## Methods Implemented

| # | Method | Module | Boundary Condition | Norm Conservation | Best For |
|---|--------|--------|--------------------|-------------------|----------|
| 1 | **Crank–Nicolson** | `CrankNicolson` | Zero (Dirichlet) | Exact | General 1-D problems |
| 2 | **4th-order Runge–Kutta** | `RungeKutta` | Zero (Dirichlet) | Approximate | Benchmarking / small Δt |
| 3 | **Chebyshev Expansion** | `ChebyshevPoly` | Zero (Dirichlet) | Near-exact | High-accuracy spectral work |
| 4 | **Lanczos / Krylov** | `Lanczos` | Zero (Dirichlet) | Near-exact | Large time steps (M controls accuracy) |
| 5 | **Split-Operator (FFT)** | `SplitOperatorKin` | Periodic | Exact | 2-D / 3-D; unconditionally stable |

---

## Repository Structure

```
Schrödinger_Equation/
│
├── src/                         # Core library
│   ├── __init__.py              # Package exports
│   ├── tdse_solvers.py          # Five propagation methods
│   ├── wavepacket.py            # Gaussian and other wavepacket constructors
│   └── potentials.py            # 1-D and 2-D potential energy functions
│
├── scripts/                     # Runnable simulation scripts
│   ├── 01_compare_timesteps.py  # Time-step convergence (Figs. 1–2)
│   ├── 02_gaussian_evolution.py # Wavepacket snapshots at k₀ = 0, 1, 5 (Figs. 3–5)
│   ├── 03_barrier_tunneling.py  # Quantum tunnelling at V₀ = 10, 100 a.u. (Fig. 7)
│   ├── 04_compare_methods.py    # All-method comparison + norm history (Figs. 8–11)
│   ├── 05_animations.py         # Animated GIFs for all figures (Figs. 6–11)
│   └── 06_split_operator_2d.py  # 2-D free particle and double-slit
│
├── docs/
│   └── theory.md                # Full mathematical derivations and method comparison
│
├── figures/                     # Auto-generated static plots (created on run)
├── animations/                  # Auto-generated GIF animations (created on run)
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Installation

**Prerequisites:** Python 3.9 or later.

```bash
# 1. Clone the repository
git clone https://github.com/Om-Physics/Schr-dinger_Equation.git
cd Schr-dinger_Equation

# 2. (Recommended) Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

```python
import numpy as np
from src.tdse_solvers import CrankNicolson
from src.wavepacket  import gaussian_wavepacket
from src.potentials  import finite_barrier

# Define a spatial grid: x ∈ [−10, 10] a.u., 1001 points
x    = np.linspace(-10, 10, 1001)

# Initial Gaussian wavepacket: centred at −4, momentum k₀ = 10 a.u.
psi0 = gaussian_wavepacket(x, x0=-4.0, k0=10.0, sigma=0.5)

# Finite barrier at x = 0 (width = 2 a.u., height = 10 a.u.)
V    = finite_barrier(x, x_centre=0.0, width=2.0, height=10.0)

# Propagate for 600 steps with Δt = 0.0003 a.u.
PSI  = CrankNicolson(psi0, V, x, dt=0.0003, N=600)
# PSI has shape (1001, 600) — columns are wavefunctions at each time step

# Check norm at final step
norm = np.trapz(np.abs(PSI[:, -1])**2, x)
print(f"Norm at final step: {norm:.8f}")   # should be ≈ 1.0
```

---

## Scripts and Figures

Run any script from the repository root. All output files are saved automatically.

```bash
python scripts/01_compare_timesteps.py   # → figures/01a_*.png, 01b_*.png
python scripts/02_gaussian_evolution.py  # → figures/02a_*.png, 02b_*.png, 02c_*.png
python scripts/03_barrier_tunneling.py   # → figures/03a_*.png … 03c_*.png
python scripts/04_compare_methods.py     # → figures/04a_*.png … 04e_*.png
python scripts/05_animations.py          # → animations/05a_*.gif … 05f_*.gif
python scripts/06_split_operator_2d.py  # → figures/06a_*.png, 06b_*.png + animations
```

### Figure descriptions

**Script 01 — Time-step convergence (Crank–Nicolson)**  
Computes the CN wavefunction at $t = 1.0$ a.u. with three different time steps and plots both the probability density and the pairwise differences. Demonstrates that for $k_0 = 0$, $\Delta t = 0.01$ already gives reasonable results, while for $k_0 = 10$ a much smaller $\Delta t \approx 0.003$ is needed.

**Script 02 — Wavepacket evolution snapshots**  
Plots $|\psi(x, t)|^2$ at six evenly-spaced times for $k_0 = 0$, $1$, and $5$ a.u. Shows the interplay of spreading (dispersion) and translation.

**Script 03 — Quantum tunnelling**  
A wavepacket with $k_0 = 10$ a.u. is incident on barriers of height $V_0 = 10$ and $V_0 = 100$ a.u. (width $= 2$ a.u.). The transmission coefficient $T$ is printed to stdout. Tunnelling is clearly visible at $V_0 = 10$ and negligible at $V_0 = 100$.

**Script 04 — All-method comparison + norm conservation**  
Runs all five methods on the same initial condition and compares pairwise against the CN reference. Also plots the norm $\langle\psi|\psi\rangle$ as a function of time for all methods, illustrating that RK4 deviates from unity at large $\Delta t$.

**Script 05 — Animated GIFs**  
Produces six animations corresponding directly to Figures 6–11 of the source post.

**Script 06 — 2-D Split-Operator**  
Demonstrates 2-D propagation of a Gaussian wavepacket in free space and through a double-slit barrier, producing both static snapshots and animated GIFs.

---

## Theory Summary

Full derivations are in [`docs/theory.md`](docs/theory.md). The key equations are summarised below.

### Crank–Nicolson (Cayley form)

$$
\left(\mathbb{I} + \tfrac{i\Delta t}{2}\mathcal{H}\right)\Psi^{n+1}
= \left(\mathbb{I} - \tfrac{i\Delta t}{2}\mathcal{H}\right)\Psi^n
$$

### 4th-Order Runge–Kutta

$$
\Psi^{n+1} = \Psi^n + \tfrac{1}{6}(k_1 + 2k_2 + 2k_3 + k_4), \quad
k_i \text{ defined by } f = -i\mathcal{H}\Psi
$$

### Chebyshev Expansion

$$
e^{-i\mathcal{H}\Delta t}
= J_0(\tau)\,\mathbb{I} + 2\sum_{n=1}^{N}(-i)^n J_n(\tau)\,T_n(\tilde{H})
$$

where $\tau = \|\mathcal{H}\|\Delta t$, $\tilde{H} = \mathcal{H}/\|\mathcal{H}\|$, $J_n$ are Bessel functions, and $T_n$ are Chebyshev polynomials computed by the three-term recurrence.

### Lanczos / Krylov

$$
\psi(t+\Delta t) \approx V_M \cdot U\,e^{-iE^L\Delta t}\,U^\dagger e_1
$$

where $V_M$ is the matrix of $M$ orthonormal Lanczos basis vectors, and $H^L = UE^LU^\dagger$ is the reduced Hamiltonian diagonalised in the Krylov subspace.

### Split-Operator (kinetic-referenced, 2nd order)

$$
\Psi^{n+1}
= e^{-i\hat{V}\Delta t/2}\,
  \mathcal{F}^{-1}\!\left[e^{-ik^2\Delta t/2}\cdot\mathcal{F}\!\left[e^{-i\hat{V}\Delta t/2}\,\Psi^n\right]\right]
$$

### Initial Gaussian Wavepacket

$$
\psi_0(x) = \frac{1}{(\pi\sigma_x^2)^{1/4}}
\exp\!\left[i k_0(x-x_0)\right]
\exp\!\left[-\frac{(x-x_0)^2}{2\sigma_x^2}\right]
$$

---

## Results

The following animations and figures are produced by the scripts above.

### Wavepacket propagation — $k_0 = 0$ vs $k_0 = 10$ (Crank–Nicolson)
> `animations/05a_wavepacket_k0_k10.gif`

### Quantum tunnelling — $V_0 = 10$ vs $V_0 = 100$ a.u.
> `animations/05b_barrier_V10_V100.gif`

### Method comparison — CN vs Split-Operator
> `animations/05f_compare_CN_SplitOp.gif`

The key finding from the comparison is that **the two methods agree well while the wavepacket is in the interior of the domain** but diverge once it reaches the boundary — because CN enforces zero boundary conditions while Split-Operator enforces periodic ones.

### Norm conservation
> `figures/04e_norm_comparison.png`

Crank–Nicolson, Chebyshev, Lanczos, and Split-Operator all conserve the norm to machine precision. RK4 shows norm drift for larger time steps.

---

## References

1. Introduction to Quantum Mechanics, 3rd ed., [Cambridge University Press](https://www.cambridge.org/core/books/introduction-to-quantum-mechanics/990799CA07A83FC5312402AF6860311E?utm_source=chatgpt.com) (2018). ([perlego.com][1])
2. Crank, J.; Nicolson, P. (1947). *Proc. Camb. Phil. Soc.* **43**: 50–67.
3. Tal-Ezer, H.; Kosloff, R. (1984). *J. Chem. Phys.* **81**, 3967.
4. Park, T. J.; Light, J. C. (1986). *J. Chem. Phys.* **85**, 5870.
5. Hochbruck, M.; Lubich, C. (1997). *SIAM J. Numer. Anal.* **34**, 1911.
6. Feit, M. D.; Fleck, J. A.; Steiger, A. (1982). *J. Comput. Phys.* **47**, 412–433.
7. QMsolve — [github.com/quantum-visualizations/qmsolve](https://github.com/quantum-visualizations/qmsolve)
8. SciPy documentation — [docs.scipy.org](https://docs.scipy.org)

---

## License

This project is released under the [MIT License](LICENSE).
