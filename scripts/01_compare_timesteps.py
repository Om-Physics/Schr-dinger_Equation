#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
01_compare_timesteps.py
=======================
Reproduce Figures 1 & 2 from Qijing Zheng's TDSE post:

    Effect of the time step Δt on the Crank–Nicolson solution.

Two cases are studied:
    • k₀ = 0  (stationary wavepacket — Fig. 1)
    • k₀ = 10 (fast-moving wavepacket — Fig. 2)

For each case, the wavefunction at t = 1.0 a.u. is computed with three
different time steps, and the pairwise differences are plotted alongside the
probability density  |ψ(x, t)|².

Output files
------------
    figures/01a_timestep_comparison_k0.png
    figures/01b_timestep_comparison_k10.png
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Allow running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.tdse_solvers import CrankNicolson, estimate_dt
from src.wavepacket  import gaussian_wavepacket

os.makedirs("figures", exist_ok=True)

# ---------------------------------------------------------------------------
# Shared grid parameters
# ---------------------------------------------------------------------------
L      = 1.0          # domain half-length → x ∈ [−L, L]
J      = 1000         # number of interior grid points
x      = np.linspace(-L, L, J + 1)
t_end  = 1.0          # target propagation time (a.u.)
sigma  = 0.4          # wavepacket width (a.u.)
x0     = 0.0          # initial wavepacket centre

CMAP   = plt.cm.tab10

# ---------------------------------------------------------------------------
# Helper: run CN for a given k0 and list of dts, return PSI at t = t_end
# ---------------------------------------------------------------------------

def run_case(k0, dt_list, t_end=1.0):
    results = {}
    for dt in dt_list:
        N = max(int(t_end / dt) + 1, 2)
        t = np.linspace(0, (N - 1) * dt, N)
        psi0 = gaussian_wavepacket(x, x0=x0, k0=k0, sigma=sigma)
        V    = np.zeros_like(x)
        PSI  = CrankNicolson(psi0, V, x, dt=dt, N=N)
        # interpolate to exactly t_end
        idx = np.argmin(np.abs(t - t_end))
        results[dt] = PSI[:, idx]
    return results


# ---------------------------------------------------------------------------
# Case 1 : k0 = 0
# ---------------------------------------------------------------------------
print("Case 1: k₀ = 0")
k0_A    = 0
dt_rec  = estimate_dt(k0_A, sigma)
print(f"  Recommended Δt ≈ {dt_rec:.4f} a.u.")

dts_A = [0.1, 0.01, 0.001]
res_A = run_case(k0_A, dts_A)
ref_A = res_A[0.001]   # finest step is "reference"

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
fig.suptitle(r"Fig. 1 — Time-step convergence: $k_0 = 0$, $\sigma_x = 0.4$ a.u.",
             fontsize=13, fontweight='bold')

ax = axes[0]
for i, dt in enumerate(dts_A):
    psi = res_A[dt]
    ax.plot(x, np.abs(psi)**2, color=CMAP(i),
            label=fr"$\Delta t = {dt}$", linewidth=1.8)
ax.set_xlabel(r"$x$ (a.u.)", fontsize=12)
ax.set_ylabel(r"$|\psi(x,\, t{=}1)|^2$", fontsize=12)
ax.set_title("Probability density at $t = 1.0$ a.u.")
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

ax = axes[1]
for i, dt in enumerate(dts_A[:-1]):   # differences vs finest
    diff = np.abs(res_A[dt] - ref_A)
    ax.plot(x, diff, color=CMAP(i),
            label=fr"$|\psi_{{\Delta t={dt}}} - \psi_{{\Delta t=0.001}}|$",
            linewidth=1.8)
ax.set_xlabel(r"$x$ (a.u.)", fontsize=12)
ax.set_ylabel("Absolute difference", fontsize=12)
ax.set_title(r"Difference vs. reference ($\Delta t = 0.001$)")
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("figures/01a_timestep_comparison_k0.png", dpi=150)
plt.close()
print("  Saved: figures/01a_timestep_comparison_k0.png")

# ---------------------------------------------------------------------------
# Case 2 : k0 = 10
# ---------------------------------------------------------------------------
print("Case 2: k₀ = 10")
k0_B   = 10
dt_rec = estimate_dt(k0_B, sigma)
print(f"  Recommended Δt ≈ {dt_rec:.4f} a.u.")

dts_B = [0.01, 0.003, 0.0005]
res_B = run_case(k0_B, dts_B, t_end=0.3)   # shorter propagation for fast packet
ref_B = res_B[0.0005]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
fig.suptitle(r"Fig. 2 — Time-step convergence: $k_0 = 10$, $\sigma_x = 0.4$ a.u.",
             fontsize=13, fontweight='bold')

ax = axes[0]
for i, dt in enumerate(dts_B):
    psi = res_B[dt]
    ax.plot(x, np.abs(psi)**2, color=CMAP(i),
            label=fr"$\Delta t = {dt}$", linewidth=1.8)
ax.set_xlabel(r"$x$ (a.u.)", fontsize=12)
ax.set_ylabel(r"$|\psi(x,\, t)|^2$", fontsize=12)
ax.set_title(r"Probability density at $t = 0.3$ a.u.")
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

ax = axes[1]
for i, dt in enumerate(dts_B[:-1]):
    diff = np.abs(res_B[dt] - ref_B)
    ax.plot(x, diff, color=CMAP(i),
            label=fr"$|\psi_{{\Delta t={dt}}} - \psi_\mathrm{{ref}}|$",
            linewidth=1.8)
ax.set_xlabel(r"$x$ (a.u.)", fontsize=12)
ax.set_ylabel("Absolute difference", fontsize=12)
ax.set_title(r"Difference vs. reference ($\Delta t = 0.0005$)")
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("figures/01b_timestep_comparison_k10.png", dpi=150)
plt.close()
print("  Saved: figures/01b_timestep_comparison_k10.png")

print("\n[Script 01 complete]")
