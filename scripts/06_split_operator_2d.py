#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
06_split_operator_2d.py
=======================
Demonstrate the Split-Operator method on two canonical 2-D problems:

    (A) Free Gaussian wavepacket in a 2-D infinite well
        — shows isotropic spreading and reflections from walls.

    (B) Double-slit interference
        — a 2-D Gaussian wavepacket propagates through a double-slit barrier;
          the interference pattern on the far side is computed and plotted.

Output files
------------
    figures/06a_2d_free_snapshots.png   — probability density at 4 times
    figures/06b_2d_doubleslit.png       — probability density after the slits
    animations/06c_2d_free.gif          — animated propagation (free)
    animations/06d_2d_doubleslit.gif    — animated double-slit propagation
"""

import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.tdse_solvers import SplitOperatorKin
from src.wavepacket  import gaussian_wavepacket_2d
from src.potentials  import double_slit_2d

os.makedirs("figures",    exist_ok=True)
os.makedirs("animations", exist_ok=True)

FPS      = 25
WRITER   = "pillow"
INTERVAL = 50

# ---------------------------------------------------------------------------
# Common 2-D grid
# ---------------------------------------------------------------------------
Lx, Ly   = 8.0, 8.0
Nx, Ny   = 256, 256
x1d      = np.linspace(-Lx/2, Lx/2, Nx)
y1d      = np.linspace(-Ly/2, Ly/2, Ny)
X, Y     = np.meshgrid(x1d, y1d, indexing='ij')   # shape (Nx, Ny)

# ---------------------------------------------------------------------------
# (A) Free particle in 2-D box
# ---------------------------------------------------------------------------
print("2-D free wavepacket ...")

kx0, ky0 = 3.0, 2.0
sx, sy   = 0.5, 0.5
psi0_A   = gaussian_wavepacket_2d(X, Y, x0=-1.0, y0=-1.0,
                                   kx=kx0, ky=ky0,
                                   sigma_x=sx, sigma_y=sy)
V_A      = np.zeros((Nx, Ny))
dt_A     = 0.02
N_A      = 80

PSI_A = SplitOperatorKin(psi0_A, V_A, [Lx, Ly], dt=dt_A, N=N_A)

# Static plot: 4 snapshots
snap_idx = [0, 20, 40, 70]
fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
fig.suptitle(r"Fig. A — 2-D Gaussian wavepacket in free space (periodic BC), "
             fr"$k_x={kx0}$, $k_y={ky0}$, $\sigma={sx}$ a.u.",
             fontsize=11, fontweight='bold')
for ax, idx in zip(axes, snap_idx):
    prob = np.abs(PSI_A[:, :, idx])**2
    im = ax.pcolormesh(X, Y, prob, cmap='inferno', shading='auto')
    ax.set_title(fr"$t = {idx * dt_A:.2f}$ a.u.", fontsize=10)
    ax.set_xlabel(r"$x$ (a.u.)", fontsize=9)
    ax.set_ylabel(r"$y$ (a.u.)", fontsize=9)
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig("figures/06a_2d_free_snapshots.png", dpi=130)
plt.close()
print("  Saved: figures/06a_2d_free_snapshots.png")

# Animated GIF
fig, ax = plt.subplots(figsize=(5, 5))
im_ani = ax.pcolormesh(X, Y, np.zeros((Nx, Ny)), cmap='inferno',
                        shading='auto', vmin=0, vmax=(np.abs(PSI_A[:, :, 0])**2).max())
ax.set_xlabel(r"$x$ (a.u.)", fontsize=11)
ax.set_ylabel(r"$y$ (a.u.)", fontsize=11)
ax.set_aspect('equal')
ttl = ax.set_title("", fontsize=11)
plt.colorbar(im_ani, ax=ax, fraction=0.046, pad=0.04, label=r"$|\psi|^2$")
plt.tight_layout()

def _upd_A(frame):
    prob = np.abs(PSI_A[:, :, frame])**2
    im_ani.set_array(prob.ravel())
    ttl.set_text(fr"2-D free wavepacket — $t = {frame * dt_A:.2f}$ a.u.")
    return im_ani,

ani_A = animation.FuncAnimation(fig, _upd_A, frames=N_A, interval=INTERVAL, blit=True)
ani_A.save("animations/06c_2d_free.gif", writer=WRITER, fps=FPS)
plt.close(fig)
print("  Saved: animations/06c_2d_free.gif")


# ---------------------------------------------------------------------------
# (B) Double-slit interference
# ---------------------------------------------------------------------------
print("2-D double-slit ...")

psi0_B = gaussian_wavepacket_2d(X, Y, x0=-2.5, y0=0.0,
                                  kx=5.0, ky=0.0,
                                  sigma_x=0.5, sigma_y=0.8)
V_B    = double_slit_2d(X, Y, x_wall=0.0, wall_thickness=0.15,
                         slit_half_width=0.35, slit_separation=1.4,
                         height=2000.0)
dt_B   = 0.006
N_B    = 120

PSI_B = SplitOperatorKin(psi0_B, V_B, [Lx, Ly], dt=dt_B, N=N_B)

# Static plot: initial + 3 later times
snap_idx_B = [0, 40, 80, 119]
fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
fig.suptitle(r"Fig. B — 2-D double-slit interference ($k_x=5$, $\sigma_x=0.5$, "
             r"slit separation $= 1.4$ a.u.)",
             fontsize=11, fontweight='bold')
vmax_B = (np.abs(PSI_B[:, :, 0])**2).max() * 0.4
for ax, idx in zip(axes, snap_idx_B):
    prob = np.abs(PSI_B[:, :, idx])**2
    # Overlay barrier (gray)
    wall_mask = V_B > 0
    barrier_overlay = np.where(wall_mask, 0.8 * vmax_B, np.nan)
    im = ax.pcolormesh(X, Y, prob, cmap='hot', shading='auto', vmin=0, vmax=vmax_B)
    ax.pcolormesh(X, Y, barrier_overlay, cmap='Greys', shading='auto',
                   vmin=0, vmax=vmax_B, alpha=0.5)
    ax.set_title(fr"$t = {idx * dt_B:.3f}$ a.u.", fontsize=10)
    ax.set_xlabel(r"$x$ (a.u.)", fontsize=9)
    ax.set_ylabel(r"$y$ (a.u.)", fontsize=9)
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig("figures/06b_2d_doubleslit.png", dpi=130)
plt.close()
print("  Saved: figures/06b_2d_doubleslit.png")

# Animated GIF
barrier_overlay_B = np.where(V_B > 0, vmax_B, 0.0)
fig, ax = plt.subplots(figsize=(5, 5))
im_B = ax.pcolormesh(X, Y, np.zeros((Nx, Ny)), cmap='hot',
                      shading='auto', vmin=0, vmax=vmax_B)
ax.pcolormesh(X, Y, barrier_overlay_B, cmap='Greys', shading='auto',
               vmin=0, vmax=vmax_B, alpha=0.5)
ax.set_xlabel(r"$x$ (a.u.)", fontsize=11)
ax.set_ylabel(r"$y$ (a.u.)", fontsize=11)
ax.set_aspect('equal')
ttl_B = ax.set_title("", fontsize=11)
plt.colorbar(im_B, ax=ax, fraction=0.046, pad=0.04, label=r"$|\psi|^2$")
plt.tight_layout()

def _upd_B(frame):
    prob = np.abs(PSI_B[:, :, frame])**2
    im_B.set_array(prob.ravel())
    ttl_B.set_text(fr"Double-slit — $t = {frame * dt_B:.3f}$ a.u.")
    return im_B,

ani_B = animation.FuncAnimation(fig, _upd_B, frames=N_B, interval=INTERVAL, blit=True)
ani_B.save("animations/06d_2d_doubleslit.gif", writer=WRITER, fps=FPS)
plt.close(fig)
print("  Saved: animations/06d_2d_doubleslit.gif")

print("\n[Script 06 complete]")
