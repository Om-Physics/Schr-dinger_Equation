#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
03_barrier_tunneling.py
=======================
Reproduce Figure 7 from Qijing Zheng's TDSE post.

A Gaussian wavepacket with average momentum k₀ = 10 a.u. propagates toward
a finite rectangular barrier of width 2 a.u. and two different heights:

    • V₀ = 10  a.u.  — modest barrier; quantum tunnelling is visible
    • V₀ = 100 a.u.  — tall barrier; tunnelling is negligible

For each case, the probability density is shown at several time snapshots.

The transmission coefficient T is also printed to stdout for both cases.

Output files
------------
    figures/03a_tunneling_V10.png
    figures/03b_tunneling_V100.png
    figures/03c_tunneling_comparison.png
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.tdse_solvers import CrankNicolson
from src.wavepacket  import gaussian_wavepacket
from src.potentials  import finite_barrier

os.makedirs("figures", exist_ok=True)

# ---------------------------------------------------------------------------
# Grid and wavepacket parameters
# ---------------------------------------------------------------------------
L          = 20.0
J          = 2000
x          = np.linspace(-L/2, L/2, J + 1)
k0         = 10.0      # a.u. — wavepacket momentum
sigma      = 0.5       # a.u. — wavepacket width
x0         = -4.0      # a.u. — initial centre (left of barrier)
dt         = 0.0003    # a.u.
t_end      = 0.8       # a.u.
N          = int(t_end / dt) + 2
t          = np.linspace(0, (N-1)*dt, N)

barrier_x  = 0.0       # barrier centre
barrier_w  = 2.0       # barrier width (a.u.)

NSNAPS     = 5
CMAP_SEQ   = cm.plasma


def transmission_coefficient(PSI, x, barrier_x, barrier_w):
    """Fraction of probability density to the right of the barrier."""
    x_right = x > (barrier_x + barrier_w / 2.0)
    prob_total = np.trapezoid(np.abs(PSI[:, -1])**2, x)
    prob_right = np.trapezoid(np.abs(PSI[x_right, -1])**2, x[x_right])
    return prob_right / prob_total


# ---------------------------------------------------------------------------
# Run both barrier heights
# ---------------------------------------------------------------------------

cases = [
    {"V0": 10,  "filename": "figures/03a_tunneling_V10.png",  "label": "Fig. 7a"},
    {"V0": 100, "filename": "figures/03b_tunneling_V100.png", "label": "Fig. 7b"},
]

results = {}
for case in cases:
    V0   = case["V0"]
    V    = finite_barrier(x, x_centre=barrier_x, width=barrier_w, height=V0)
    psi0 = gaussian_wavepacket(x, x0=x0, k0=k0, sigma=sigma)
    print(f"Running V₀ = {V0} a.u. ...")
    PSI = CrankNicolson(psi0, V, x, dt=dt, N=N)
    results[V0] = PSI

    T = transmission_coefficient(PSI, x, barrier_x, barrier_w)
    print(f"  Transmission coefficient T ≈ {T:.4f}")

    snap_idx = np.linspace(0, N-1, NSNAPS, dtype=int)
    colors   = [CMAP_SEQ(v) for v in np.linspace(0.1, 0.95, NSNAPS)]

    fig, ax = plt.subplots(figsize=(11, 5))
    # shade barrier
    ax.axvspan(barrier_x - barrier_w/2, barrier_x + barrier_w/2,
               alpha=0.25, color='gray', label=fr"Barrier $V_0={V0}$ a.u.")
    for ii, idx in enumerate(snap_idx):
        prob = np.abs(PSI[:, idx])**2
        ax.plot(x, prob, color=colors[ii],
                label=fr"$t = {t[idx]:.3f}$ a.u.", linewidth=1.8)

    ax.set_xlabel(r"$x$ (a.u.)", fontsize=13)
    ax.set_ylabel(r"$|\psi(x,\,t)|^2$", fontsize=13)
    ax.set_title(
        fr"{case['label']} — Tunnelling through $V_0={V0}$ a.u. barrier, "
        fr"$k_0={k0}$, $\sigma_x={sigma}$ a.u. | $T \approx {T:.3f}$",
        fontsize=11, fontweight='bold'
    )
    ax.legend(fontsize=9, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(case["filename"], dpi=150)
    plt.close()
    print(f"  Saved: {case['filename']}")


# ---------------------------------------------------------------------------
# Side-by-side comparison at the final time step
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
fig.suptitle(
    fr"Fig. 7 — Quantum tunnelling comparison: $k_0={k0}$, $\sigma_x={sigma}$ a.u., "
    fr"$t_\mathrm{{end}} = {t_end}$ a.u.",
    fontsize=12, fontweight='bold'
)
for ax, (V0, PSI) in zip(axes, results.items()):
    ax.axvspan(barrier_x - barrier_w/2, barrier_x + barrier_w/2,
               alpha=0.30, color='gray', label=fr"$V_0 = {V0}$ a.u.")
    ax.plot(x, np.abs(PSI[:, -1])**2, 'steelblue', linewidth=2,
            label=r"$|\psi(x,t_\mathrm{end})|^2$")
    T = transmission_coefficient(PSI, x, barrier_x, barrier_w)
    ax.set_title(fr"$V_0 = {V0}$ a.u. — $T \approx {T:.3f}$", fontsize=12)
    ax.set_xlabel(r"$x$ (a.u.)", fontsize=12)
    ax.set_ylabel(r"$|\psi|^2$", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("figures/03c_tunneling_comparison.png", dpi=150)
plt.close()
print("  Saved: figures/03c_tunneling_comparison.png")

print("\n[Script 03 complete]")
