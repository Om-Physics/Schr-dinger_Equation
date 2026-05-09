#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
02_gaussian_evolution.py
========================
Reproduce Figures 3, 4, and 5 from Qijing Zheng's TDSE post.

The probability density |ψ(x, t)|² is plotted at several time snapshots
for three values of the initial momentum:

    • k₀ = 0  (stationary, spreading wavepacket — Fig. 3)
    • k₀ = 1  (slow-moving wavepacket — Fig. 4)
    • k₀ = 5  (fast-moving wavepacket — Fig. 5)

Method : Crank–Nicolson with zero boundary conditions.
Domain : x ∈ [−L/2, L/2] (effectively an infinite square well).

Output files
------------
    figures/02a_evolution_k0.png
    figures/02b_evolution_k1.png
    figures/02c_evolution_k5.png
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.tdse_solvers import CrankNicolson, estimate_dt
from src.wavepacket  import gaussian_wavepacket

os.makedirs("figures", exist_ok=True)

# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
L     = 10.0
J     = 800
x     = np.linspace(-L/2, L/2, J + 1)
x0    = 0.0
sigma = 0.4

NSNAPS   = 6          # number of time snapshots per plot
CMAP_SEQ = cm.viridis  # sequential colormap for time progression


# ---------------------------------------------------------------------------
# Helper: run and plot a single k₀ case
# ---------------------------------------------------------------------------

def run_and_plot(k0, dt, t_end, filename, fig_label):
    N    = max(int(t_end / dt) + 2, 2)
    t    = np.linspace(0, (N - 1) * dt, N)
    psi0 = gaussian_wavepacket(x, x0=x0, k0=k0, sigma=sigma)
    V    = np.zeros_like(x)
    PSI  = CrankNicolson(psi0, V, x, dt=dt, N=N)

    # Select evenly-spaced snapshot indices
    snap_indices = np.linspace(0, N - 1, NSNAPS, dtype=int)
    colors       = [CMAP_SEQ(v) for v in np.linspace(0.1, 0.95, NSNAPS)]

    fig, ax = plt.subplots(figsize=(10, 5))
    for ii, idx in enumerate(snap_indices):
        prob = np.abs(PSI[:, idx])**2
        ax.plot(x, prob, color=colors[ii],
                label=fr"$t = {t[idx]:.3f}$ a.u.", linewidth=1.8)

    ax.set_xlabel(r"$x$ (a.u.)", fontsize=13)
    ax.set_ylabel(r"$|\psi(x,\,t)|^2$", fontsize=13)
    ax.set_title(
        fr"{fig_label} — Gaussian wavepacket evolution, $k_0={k0}$, "
        fr"$\sigma_x={sigma}$ a.u.",
        fontsize=12, fontweight='bold'
    )
    ax.legend(fontsize=9, loc='upper right', ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Saved: {filename}")


# ---------------------------------------------------------------------------
# k₀ = 0
# ---------------------------------------------------------------------------
print("k₀ = 0")
dt_rec = estimate_dt(0, sigma)
print(f"  Recommended Δt ≈ {dt_rec:.4f}")
run_and_plot(
    k0=0, dt=0.005, t_end=4.0,
    filename="figures/02a_evolution_k0.png",
    fig_label="Fig. 3"
)

# ---------------------------------------------------------------------------
# k₀ = 1
# ---------------------------------------------------------------------------
print("k₀ = 1")
dt_rec = estimate_dt(1, sigma)
print(f"  Recommended Δt ≈ {dt_rec:.4f}")
run_and_plot(
    k0=1, dt=0.003, t_end=3.0,
    filename="figures/02b_evolution_k1.png",
    fig_label="Fig. 4"
)

# ---------------------------------------------------------------------------
# k₀ = 5
# ---------------------------------------------------------------------------
print("k₀ = 5")
dt_rec = estimate_dt(5, sigma)
print(f"  Recommended Δt ≈ {dt_rec:.4f}")
run_and_plot(
    k0=5, dt=0.001, t_end=1.0,
    filename="figures/02c_evolution_k5.png",
    fig_label="Fig. 5"
)

print("\n[Script 02 complete]")
