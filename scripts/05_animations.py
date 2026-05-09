#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
05_animations.py
================
Generate animated GIFs and MP4s corresponding to Figures 6–11 from the TDSE post.

Animations produced
-------------------
    animations/05a_wavepacket_k0_k10.gif    — Fig. 6: two k₀ values side-by-side (CN)
    animations/05b_barrier_V10_V100.gif     — Fig. 7: tunneling, low vs high barrier
    animations/05c_compare_CN_RK4.gif       — Fig. 8: CN vs RK4
    animations/05d_compare_CN_Chebyshev.gif — Fig. 9: CN vs Chebyshev
    animations/05e_compare_CN_Lanczos.gif   — Fig. 10: CN vs Lanczos
    animations/05f_compare_CN_SplitOp.gif   — Fig. 11: CN vs Split-Operator

Requirements: matplotlib (with Pillow for GIF writer), numpy, scipy.
"""

import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")            # headless backend
import matplotlib.pyplot as plt
import matplotlib.animation as animation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.tdse_solvers import (CrankNicolson, RungeKutta,
                               ChebyshevPoly, Lanczos,
                               SplitOperatorKin)
from src.wavepacket  import gaussian_wavepacket
from src.potentials  import zero_potential, finite_barrier

os.makedirs("animations", exist_ok=True)

WRITER   = "pillow"   # for .gif output; change to "ffmpeg" for .mp4
FPS      = 30
INTERVAL = 40         # ms per frame

# ---------------------------------------------------------------------------
# Helper: thin a trajectory to at most MAX_FRAMES frames
# ---------------------------------------------------------------------------
MAX_FRAMES = 200

def thin(PSI, max_frames=MAX_FRAMES):
    N  = PSI.shape[1]
    if N <= max_frames:
        return PSI
    idx = np.linspace(0, N-1, max_frames, dtype=int)
    return PSI[:, idx]


# ===========================================================================
# Animation A — Fig. 6: k₀ = 0 and k₀ = 10 side by side
# ===========================================================================
print("Animation A — k₀ = 0 vs k₀ = 10 ...")

L_A, J_A = 20.0, 1000
x_A = np.linspace(-L_A/2, L_A/2, J_A + 1)
sigma_A = 0.4
dt_A    = 0.001
N_A     = 600

psi0_k0  = gaussian_wavepacket(x_A, x0=0.0, k0=0,  sigma=sigma_A)
psi0_k10 = gaussian_wavepacket(x_A, x0=0.0, k0=10, sigma=sigma_A)
V_A      = zero_potential(x_A)

PSI_k0  = thin(CrankNicolson(psi0_k0,  V_A, x_A, dt=dt_A, N=N_A))
PSI_k10 = thin(CrankNicolson(psi0_k10, V_A, x_A, dt=0.0003, N=N_A))
NF = min(PSI_k0.shape[1], PSI_k10.shape[1])

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
fig.suptitle(r"Gaussian wavepacket: $k_0=0$ (left) vs $k_0=10$ a.u. (right)",
             fontsize=12, fontweight='bold')
lines = []
for ax, k0_label in zip(axes, ["$k_0 = 0$", "$k_0 = 10$ a.u."]):
    ln, = ax.plot([], [], 'C0', lw=2)
    fill = ax.fill_between(x_A, 0, np.zeros_like(x_A), alpha=0.35, color='C0')
    ax.set_xlim(x_A[0], x_A[-1])
    ax.set_ylim(0, 0.8)
    ax.set_xlabel(r"$x$ (a.u.)", fontsize=12)
    ax.set_ylabel(r"$|\psi|^2$", fontsize=12)
    ax.set_title(k0_label, fontsize=12)
    ax.grid(alpha=0.3)
    lines.append((ax, ln))

time_text = fig.text(0.5, 0.01, "", ha='center', fontsize=11)

def _update_A(frame):
    for (ax, ln), PSI in zip(lines, [PSI_k0, PSI_k10]):
        prob = np.abs(PSI[:, frame])**2
        ln.set_data(x_A, prob)
        # redraw fill
        for coll in ax.collections:
            coll.remove()
        ax.fill_between(x_A, 0, prob, alpha=0.30, color='C0')
    time_text.set_text(fr"$t = {frame * dt_A:.3f}$ a.u.")
    return []

ani_A = animation.FuncAnimation(fig, _update_A, frames=NF, interval=INTERVAL, blit=False)
ani_A.save("animations/05a_wavepacket_k0_k10.gif", writer=WRITER, fps=FPS)
plt.close(fig)
print("  Saved: animations/05a_wavepacket_k0_k10.gif")


# ===========================================================================
# Animation B — Fig. 7: barrier tunneling V₀=10 vs V₀=100
# ===========================================================================
print("Animation B — tunneling V₀=10 vs V₀=100 ...")

L_B, J_B = 20.0, 2000
x_B  = np.linspace(-L_B/2, L_B/2, J_B + 1)
k0_B, sigma_B, x0_B = 10.0, 0.5, -4.0
dt_B = 0.0003
N_B  = int(0.8 / dt_B) + 2

psi0_B = gaussian_wavepacket(x_B, x0=x0_B, k0=k0_B, sigma=sigma_B)
V10  = finite_barrier(x_B, x_centre=0.0, width=2.0, height=10)
V100 = finite_barrier(x_B, x_centre=0.0, width=2.0, height=100)

PSI_V10  = thin(CrankNicolson(psi0_B, V10,  x_B, dt=dt_B, N=N_B))
PSI_V100 = thin(CrankNicolson(psi0_B, V100, x_B, dt=dt_B, N=N_B))
NF_B = min(PSI_V10.shape[1], PSI_V100.shape[1])

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
fig.suptitle(r"Quantum tunnelling: $V_0=10$ a.u. (left) vs $V_0=100$ a.u. (right)",
             fontsize=12, fontweight='bold')

lines_B = []
for ax, V_plot, V0_label in zip(axes,
        [V10, V100], ["$V_0 = 10$ a.u.", "$V_0 = 100$ a.u."]):
    V_norm = V_plot / V_plot.max()
    ax.fill_between(x_B, 0, 0.2 * V_norm, color='gray', alpha=0.3, label="Barrier")
    ln, = ax.plot([], [], 'C1', lw=2)
    ax.set_xlim(-7, 7)
    ax.set_ylim(0, 0.6)
    ax.set_xlabel(r"$x$ (a.u.)", fontsize=12)
    ax.set_ylabel(r"$|\psi|^2$", fontsize=12)
    ax.set_title(V0_label, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    lines_B.append((ax, ln))

time_text_B = fig.text(0.5, 0.01, "", ha='center', fontsize=11)

def _update_B(frame):
    for (ax, ln), PSI in zip(lines_B, [PSI_V10, PSI_V100]):
        prob = np.abs(PSI[:, frame])**2
        ln.set_data(x_B, prob)
    time_text_B.set_text(fr"$t = {frame * dt_B:.4f}$ a.u.")
    return []

ani_B = animation.FuncAnimation(fig, _update_B, frames=NF_B, interval=INTERVAL, blit=False)
ani_B.save("animations/05b_barrier_V10_V100.gif", writer=WRITER, fps=FPS)
plt.close(fig)
print("  Saved: animations/05b_barrier_V10_V100.gif")


# ===========================================================================
# Helper: generic side-by-side comparison animation
# ===========================================================================

def comparison_animation(x_grid, PSI_ref, PSI_alt,
                          label_ref, label_alt, title, filename,
                          dt_ref, dt_alt=None):
    if dt_alt is None:
        dt_alt = dt_ref
    NF = min(PSI_ref.shape[1], PSI_alt.shape[1])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle(title, fontsize=11, fontweight='bold')

    ymax = max(float((np.abs(PSI_ref[:, 0])**2).max()), 0.5) * 2.2

    def _setup(ax, label):
        ax.set_xlim(x_grid[0], x_grid[-1])
        ax.set_ylim(0, ymax)
        ax.set_xlabel(r"$x$ (a.u.)", fontsize=12)
        ax.set_ylabel(r"$|\psi|^2$", fontsize=12)
        ax.set_title(label, fontsize=12)
        ax.grid(alpha=0.3)
        ln, = ax.plot([], [], lw=2)
        return ln

    ln_ref = _setup(axes[0], label_ref)
    ln_alt = _setup(axes[1], label_alt)
    time_txt = fig.text(0.5, 0.01, "", ha='center', fontsize=11)

    def _update(frame):
        prob_ref = np.abs(PSI_ref[:, min(frame, PSI_ref.shape[1]-1)])**2
        prob_alt = np.abs(PSI_alt[:, min(frame, PSI_alt.shape[1]-1)])**2
        ln_ref.set_data(x_grid, prob_ref)
        ln_alt.set_data(x_grid, prob_alt)
        time_txt.set_text(fr"$t \approx {frame * dt_ref:.4f}$ a.u.")
        return ln_ref, ln_alt

    ani = animation.FuncAnimation(fig, _update, frames=NF, interval=INTERVAL, blit=True)
    ani.save(filename, writer=WRITER, fps=FPS)
    plt.close(fig)
    print(f"  Saved: {filename}")


# ===========================================================================
# Shared setup for method comparisons
# ===========================================================================
L_C, J_C = 10.0, 800
x_C    = np.linspace(-L_C/2, L_C/2, J_C + 1)
k0_C   = 5.0
sigma_C = 0.4
psi0_C = gaussian_wavepacket(x_C, x0=0.0, k0=k0_C, sigma=sigma_C)
V_C    = zero_potential(x_C)
t_end_C = 0.5

# Reference CN
dt_CN_C = 0.0005
N_CN_C  = int(t_end_C / dt_CN_C) + 2
print("Animation C-F: running Crank-Nicolson reference ...")
PSI_CN_C = thin(CrankNicolson(psi0_C, V_C, x_C, dt=dt_CN_C, N=N_CN_C))

# ===========================================================================
# Animation C — CN vs RK4
# ===========================================================================
print("Animation C — CN vs RK4 ...")
dt_RK4_C = 0.0001
N_RK4_C  = int(t_end_C / dt_RK4_C) + 2
PSI_RK4_C = thin(RungeKutta(psi0_C, V_C, x_C, dt=dt_RK4_C, N=N_RK4_C))

comparison_animation(
    x_C, PSI_CN_C, PSI_RK4_C,
    f"Crank–Nicolson (Δt={dt_CN_C})",
    f"RK4 (Δt={dt_RK4_C})",
    fr"Fig. 8 — CN vs. RK4 | $k_0={k0_C}$, $\sigma_x={sigma_C}$ a.u.",
    "animations/05c_compare_CN_RK4.gif",
    dt_ref=dt_CN_C
)

# ===========================================================================
# Animation D — CN vs Chebyshev
# ===========================================================================
print("Animation D — CN vs Chebyshev ...")
dt_Chb_C = 0.0005
N_Chb_C  = int(t_end_C / dt_Chb_C) + 2
PSI_Chb_C = thin(ChebyshevPoly(psi0_C, V_C, x_C, dt=dt_Chb_C, N=N_Chb_C))

comparison_animation(
    x_C, PSI_CN_C, PSI_Chb_C,
    f"Crank–Nicolson (Δt={dt_Chb_C})",
    "Chebyshev (N auto)",
    fr"Fig. 9 — CN vs. Chebyshev | $k_0={k0_C}$, $\sigma_x={sigma_C}$ a.u.",
    "animations/05d_compare_CN_Chebyshev.gif",
    dt_ref=dt_CN_C
)

# ===========================================================================
# Animation E — CN vs Lanczos
# ===========================================================================
print("Animation E — CN vs Lanczos ...")
dt_Lcz_C = 0.002
N_Lcz_C  = int(t_end_C / dt_Lcz_C) + 2
PSI_Lcz_C = thin(Lanczos(psi0_C, V_C, x_C, dt=dt_Lcz_C, N=N_Lcz_C, M=15))

comparison_animation(
    x_C, PSI_CN_C, PSI_Lcz_C,
    f"Crank–Nicolson (Δt={dt_CN_C})",
    f"Lanczos M=15 (Δt={dt_Lcz_C})",
    fr"Fig. 10 — CN vs. Lanczos | $k_0={k0_C}$, $\sigma_x={sigma_C}$ a.u.",
    "animations/05e_compare_CN_Lanczos.gif",
    dt_ref=dt_CN_C
)

# ===========================================================================
# Animation F — CN vs Split-Operator
# ===========================================================================
print("Animation F — CN vs Split-Operator ...")
dt_SO_C = 0.0005
N_SO_C  = int(t_end_C / dt_SO_C) + 2
PSI_SO_C = thin(SplitOperatorKin(psi0_C, V_C, [L_C], dt=dt_SO_C, N=N_SO_C))

comparison_animation(
    x_C, PSI_CN_C, PSI_SO_C,
    f"Crank–Nicolson, zero BC (Δt={dt_CN_C})",
    f"Split-Operator, periodic BC (Δt={dt_SO_C})",
    fr"Fig. 11 — CN vs. Split-Operator | $k_0={k0_C}$, $\sigma_x={sigma_C}$ a.u.",
    "animations/05f_compare_CN_SplitOp.gif",
    dt_ref=dt_CN_C
)

print("\n[Script 05 complete — all animations saved]")
