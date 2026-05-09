#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
04_compare_methods.py
=====================
Reproduce Figures 8–11 from Qijing Zheng's TDSE post.

All four numerical methods are run on the same initial condition (Gaussian
wavepacket with k₀ = 5 a.u., σ_x = 0.4 a.u.) and compared pairwise against
the Crank–Nicolson reference:

    Fig. 8  — CN  vs. RK4
    Fig. 9  — CN  vs. Chebyshev (N = 11)
    Fig. 10 — CN  vs. Lanczos   (M = 15)
    Fig. 11 — CN  vs. Split-Operator

Each plot shows probability densities and the pointwise difference |ψ_CN − ψ_method|
at the same final propagation time, allowing direct assessment of accuracy.

Also produced:
    figures/04_norm_comparison.png  — norm conservation vs. time for all methods

Output files
------------
    figures/04a_compare_CN_RK4.png
    figures/04b_compare_CN_Chebyshev.png
    figures/04c_compare_CN_Lanczos.png
    figures/04d_compare_CN_SplitOp.png
    figures/04e_norm_comparison.png
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.tdse_solvers import (CrankNicolson, RungeKutta,
                               ChebyshevPoly, Lanczos,
                               SplitOperatorKin)
from src.wavepacket  import gaussian_wavepacket
from src.potentials  import zero_potential

os.makedirs("figures", exist_ok=True)

# ---------------------------------------------------------------------------
# Shared parameters
# ---------------------------------------------------------------------------
L      = 10.0
J      = 800
x      = np.linspace(-L/2, L/2, J + 1)
k0     = 5.0
sigma  = 0.4
x0     = 0.0
V      = zero_potential(x)
psi0   = gaussian_wavepacket(x, x0=x0, k0=k0, sigma=sigma)

# Time parameters — choose a dt fine enough for all explicit methods
dt_CN  = 0.0005
dt_RK4 = 0.0001     # RK4 needs smaller dt for same accuracy
dt_Chb = 0.0005
dt_Lcz = 0.002      # Lanczos can use larger dt with M = 15
dt_SO  = 0.0005

t_end  = 0.5
N_CN   = int(t_end / dt_CN)  + 2
N_RK4  = int(t_end / dt_RK4) + 2
N_Chb  = int(t_end / dt_Chb) + 2
N_Lcz  = int(t_end / dt_Lcz) + 2
N_SO   = int(t_end / dt_SO)  + 2

t_CN  = np.arange(N_CN)  * dt_CN
t_RK4 = np.arange(N_RK4) * dt_RK4
t_Chb = np.arange(N_Chb) * dt_Chb
t_Lcz = np.arange(N_Lcz) * dt_Lcz
t_SO  = np.arange(N_SO)  * dt_SO

# ---------------------------------------------------------------------------
# Run all methods
# ---------------------------------------------------------------------------
print("Running Crank–Nicolson ...")
PSI_CN  = CrankNicolson(psi0, V, x, dt=dt_CN,  N=N_CN)

print("Running Runge–Kutta (RK4) ...")
PSI_RK4 = RungeKutta(psi0, V, x, dt=dt_RK4, N=N_RK4)

print("Running Chebyshev expansion ...")
PSI_Chb = ChebyshevPoly(psi0, V, x, dt=dt_Chb, N=N_Chb)

print("Running Lanczos (M=15) ...")
PSI_Lcz = Lanczos(psi0, V, x, dt=dt_Lcz, N=N_Lcz, M=15)

print("Running Split-Operator ...")
PSI_SO  = SplitOperatorKin(psi0, V, [L], dt=dt_SO, N=N_SO)

# Extract wavefunction closest to t_end for each method
def snap(PSI, t_arr, t_target):
    idx = np.argmin(np.abs(t_arr - t_target))
    return PSI[:, idx], t_arr[idx]

psi_cn,  _ = snap(PSI_CN,  t_CN,  t_end)
psi_rk4, _ = snap(PSI_RK4, t_RK4, t_end)
psi_chb, _ = snap(PSI_Chb, t_Chb, t_end)
psi_lcz, _ = snap(PSI_Lcz, t_Lcz, t_end)
psi_so,  _ = snap(PSI_SO,  t_SO,  t_end)


# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------
def compare_plot(psi_ref, psi_alt, label_ref, label_alt, title, filename):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title, fontsize=12, fontweight='bold')

    ax = axes[0]
    ax.plot(x, np.abs(psi_ref)**2, 'C0', lw=2.0, label=label_ref)
    ax.plot(x, np.abs(psi_alt)**2, 'C1--', lw=1.8, label=label_alt)
    ax.set_xlabel(r"$x$ (a.u.)", fontsize=12)
    ax.set_ylabel(r"$|\psi(x,t)|^2$", fontsize=12)
    ax.set_title(fr"Probability density at $t = {t_end}$ a.u.")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    ax = axes[1]
    diff = np.abs(psi_ref - psi_alt)
    ax.semilogy(x, diff + 1e-15, 'C2', lw=1.8,
                label=fr"$|{label_ref} - {label_alt}|$")
    ax.set_xlabel(r"$x$ (a.u.)", fontsize=12)
    ax.set_ylabel("Absolute difference (log scale)", fontsize=12)
    ax.set_title("Pointwise wavefunction difference")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Saved: {filename}")


# ---------------------------------------------------------------------------
# Fig. 8 — CN vs RK4
# ---------------------------------------------------------------------------
compare_plot(
    psi_cn, psi_rk4, "CN", "RK4",
    fr"Fig. 8 — Crank–Nicolson vs. RK4 ($k_0={k0}$, $\Delta t_{{CN}}={dt_CN}$, $\Delta t_{{RK4}}={dt_RK4}$)",
    "figures/04a_compare_CN_RK4.png"
)

# ---------------------------------------------------------------------------
# Fig. 9 — CN vs Chebyshev
# ---------------------------------------------------------------------------
compare_plot(
    psi_cn, psi_chb, "CN", "Chebyshev",
    fr"Fig. 9 — Crank–Nicolson vs. Chebyshev ($k_0={k0}$, $\Delta t={dt_Chb}$)",
    "figures/04b_compare_CN_Chebyshev.png"
)

# ---------------------------------------------------------------------------
# Fig. 10 — CN vs Lanczos
# ---------------------------------------------------------------------------
compare_plot(
    psi_cn, psi_lcz, "CN", "Lanczos (M=15)",
    fr"Fig. 10 — Crank–Nicolson vs. Lanczos ($k_0={k0}$, $\Delta t_{{CN}}={dt_CN}$, $\Delta t_{{Lcz}}={dt_Lcz}$)",
    "figures/04c_compare_CN_Lanczos.png"
)

# ---------------------------------------------------------------------------
# Fig. 11 — CN vs Split-Operator
# ---------------------------------------------------------------------------
compare_plot(
    psi_cn, psi_so, "CN (zero BC)", "Split-Op (periodic BC)",
    fr"Fig. 11 — Crank–Nicolson vs. Split-Operator ($k_0={k0}$, $\Delta t={dt_SO}$)",
    "figures/04d_compare_CN_SplitOp.png"
)

# ---------------------------------------------------------------------------
# Norm conservation comparison
# ---------------------------------------------------------------------------
print("Computing norm histories ...")

def norm_history(PSI, x):
    return np.array([np.trapezoid(np.abs(PSI[:, n])**2, x) for n in range(PSI.shape[1])])

norm_cn  = norm_history(PSI_CN,  x)
norm_rk4 = norm_history(PSI_RK4, x)
norm_chb = norm_history(PSI_Chb, x)
norm_lcz = norm_history(PSI_Lcz, x)
norm_so  = norm_history(PSI_SO,  x)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(t_CN[:len(norm_cn)],   norm_cn,  'C0',   lw=2.0, label="Crank–Nicolson")
ax.plot(t_RK4[:len(norm_rk4)], norm_rk4, 'C1--', lw=1.8, label="RK4")
ax.plot(t_Chb[:len(norm_chb)], norm_chb, 'C2:',  lw=1.8, label="Chebyshev")
ax.plot(t_Lcz[:len(norm_lcz)], norm_lcz, 'C3-.',  lw=1.8, label="Lanczos")
ax.plot(t_SO[:len(norm_so)],   norm_so,  'C4',   lw=1.8, label="Split-Operator")
ax.axhline(1.0, color='k', lw=0.8, ls='--', label="Exact = 1")
ax.set_xlabel(r"$t$ (a.u.)", fontsize=13)
ax.set_ylabel(r"$\int |\psi|^2 \, dx$", fontsize=13)
ax.set_title(r"Norm conservation: $\langle\psi|\psi\rangle$ vs. time for all methods",
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.set_ylim(0.98, 1.02)
plt.tight_layout()
plt.savefig("figures/04e_norm_comparison.png", dpi=150)
plt.close()
print("  Saved: figures/04e_norm_comparison.png")

print("\n[Script 04 complete]")
