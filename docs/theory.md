# Theory: Numerical Solutions to the Time-Dependent Schrödinger Equation

> **Reference:** Introduction to Quantum Mechanics, 3rd ed., [Cambridge University Press](https://www.cambridge.org/core/books/introduction-to-quantum-mechanics/990799CA07A83FC5312402AF6860311E?utm_source=chatgpt.com) (2018). ([perlego.com][1])

[1]: https://www.perlego.com/book/4220295/introduction-to-quantum-mechanics-pdf?utm_source=chatgpt.com "[PDF] Introduction to Quantum Mechanics by David J. Griffiths, 3rd edition | 9781107189638, 9781108103145"

---

## Table of Contents

1. [The TDSE in Atomic Units](#1-the-tdse-in-atomic-units)  
2. [Discretisation of Space and Time](#2-discretisation-of-space-and-time)  
3. [Crank–Nicolson Method](#3-crankncolson-method)  
4. [4th-Order Runge–Kutta Method](#4-4th-order-rungekutta-method)  
5. [Chebyshev Polynomial Expansion](#5-chebyshev-polynomial-expansion)  
6. [Lanczos / Krylov Subspace Method](#6-lanczos--krylov-subspace-method)  
7. [Split-Operator Technique](#7-split-operator-technique)  
8. [Choosing a Time Step](#8-choosing-a-time-step)  
9. [Boundary Conditions](#9-boundary-conditions)  
10. [Method Comparison Summary](#10-method-comparison-summary)  
11. [References](#11-references)

---

## 1. The TDSE in Atomic Units

The single-particle time-dependent Schrödinger equation for an electron subject to an external potential reads:

$$
i\hbar \frac{\partial}{\partial t} \psi(\mathbf{r}, t)
= \hat{\mathcal{H}}(\mathbf{r}; t)\,\psi(\mathbf{r}, t)
= \left[-\frac{\hbar^2}{2m}\nabla^2 + \hat{V}(\mathbf{r})\right]\psi(\mathbf{r}, t)
$$

Throughout this work we adopt **Hartree atomic units** ($\hbar = m_e = e = a_0 = 1$), which simplifies the equation to:

$$
i \frac{\partial}{\partial t} \psi(\mathbf{r}, t)
= \left[-\frac{1}{2}\nabla^2 + \hat{V}(\mathbf{r})\right]\psi(\mathbf{r}, t)
$$

The **formal solution** is:

$$
\psi(\mathbf{r}, t + \Delta t) = e^{-i\mathcal{H}\,\Delta t}\,\psi(\mathbf{r}, t)
= \mathcal{U}(\Delta t)\,\psi(\mathbf{r}, t)
$$

where $\mathcal{U}(\Delta t)$ is a **unitary operator**. Unitarity guarantees that the norm of the wavefunction is conserved exactly during time propagation — a property used as a diagnostic for numerical accuracy.

### Eigenfunction expansion (time-independent Hamiltonian)

If $\mathcal{H}$ is time-independent with eigenfunctions $\phi_k$ and eigenvalues $\varepsilon_k$, any wavefunction can be expanded as:

$$
\psi(\mathbf{r}, t) = \sum_k c_k(0)\,e^{-i\varepsilon_k t}\,\phi_k(\mathbf{r})
$$

This exact solution forms the theoretical benchmark against which all numerical methods are compared.

---

## 2. Discretisation of Space and Time

The wavefunction is sampled on a uniform spatial grid with spacing $\Delta x$ and time grid with spacing $\Delta t$:

$$
\psi_j^n \equiv \psi(x_0 + j\,\Delta x,\; t_0 + n\,\Delta t)
$$

The **second spatial derivative** (kinetic operator) is approximated by the central-difference formula:

$$
\nabla^2 \psi_j^n \approx \frac{\psi_{j+1}^n + \psi_{j-1}^n - 2\psi_j^n}{(\Delta x)^2}
$$

This gives a **tridiagonal matrix representation** of $\hat{T}$:

$$
T = -\frac{1}{2\Delta x^2}
\begin{bmatrix}
-2 & 1 & 0 & \cdots \\
 1 &-2 & 1 & \cdots \\
 0 & 1 &-2 & \cdots \\
\vdots & & & \ddots
\end{bmatrix}
$$

The **potential operator** $\hat{V}$ is diagonal:

$$
V = \mathrm{diag}[v_0,\, v_1,\, v_2,\, \ldots,\, v_J]
$$

---

## 3. Crank–Nicolson Method

The Crank–Nicolson (CN) method combines the forward- and backward-Euler schemes to yield the **Cayley form** of the propagator:

$$
\mathcal{U}(\Delta t) \approx \frac{1 - \tfrac{i}{2}\mathcal{H}\Delta t}{1 + \tfrac{i}{2}\mathcal{H}\Delta t}
$$

This leads to the linear system to be solved at each time step:

$$
U_2\,\Psi^{n+1} = U_1\,\Psi^n
$$

$$
U_2 = \mathbb{I} + \tfrac{i\Delta t}{2}(T + V), \qquad
U_1 = \mathbb{I} - \tfrac{i\Delta t}{2}(T + V)
$$

**Key properties:**
- Exactly unitary — norm is conserved for any $\Delta t$.
- Second-order accurate in both space and time.
- Requires solving a sparse tridiagonal system at each step (fast via LU decomposition).
- Enforces **zero boundary conditions** implicitly.

---

## 4. 4th-Order Runge–Kutta Method

The TDSE is recast as a first-order ODE: $d\psi/dt = f(\psi) = -i\mathcal{H}\psi$.  
The classical RK4 update is:

$$
\psi^{n+1} = \psi^n + \frac{1}{6}(k_1 + 2k_2 + 2k_3 + k_4)
$$

$$
k_1 = \Delta t\,f(\psi^n), \quad
k_2 = \Delta t\,f\!\left(\psi^n + \tfrac{k_1}{2}\right), \quad
k_3 = \Delta t\,f\!\left(\psi^n + \tfrac{k_2}{2}\right), \quad
k_4 = \Delta t\,f\!\left(\psi^n + k_3\right)
$$

**Key properties:**
- 4th-order accurate in time.
- **Not unitary** — the norm will diverge for a sufficiently large $\Delta t$.
- Requires a much smaller time step than Crank–Nicolson for comparable accuracy.
- Safe guideline: $\Delta t \lesssim 1/E_\mathrm{max}$ where $E_\mathrm{max}$ is the spectral radius of $\mathcal{H}$.

---

## 5. Chebyshev Polynomial Expansion

Rather than Taylor-expanding $e^{-i\mathcal{H}\Delta t}$ (slow convergence), one expands it in **Chebyshev polynomials** $T_n$, exploiting their near-optimal approximation properties on $[-1, 1]$:

$$
e^{-i\mathcal{H}\Delta t}
= J_0(\tau)\,\mathbb{I}
  + 2\sum_{n=1}^{N} (-i)^n J_n(\tau)\,T_n(\tilde{H})
$$

where $\tilde{H} = \mathcal{H}/\|\mathcal{H}\|$ is the normalised Hamiltonian, $\tau = \|\mathcal{H}\|\,\Delta t$ is the dimensionless time, and $J_n$ are Bessel functions of the first kind.

The Chebyshev vectors $\psi_n = T_n(\tilde{H})\psi$ are built by the three-term recurrence:

$$
\psi_0 = \psi, \qquad \psi_1 = \tilde{H}\psi_0, \qquad
\psi_n = 2\tilde{H}\psi_{n-1} - \psi_{n-2}
$$

The series is truncated when $|J_n(\tau)| < \delta$ (typically $\delta \sim 10^{-10}$).

**Key properties:**
- Spectrally accurate — exponentially convergent in the number of terms.
- The truncation order $N$ depends on $\tau = \|\mathcal{H}\|\,\Delta t$; larger $\Delta t$ requires more terms.
- Effectively unitary to near machine precision.

---

## 6. Lanczos / Krylov Subspace Method

The Lanczos method builds an orthonormal basis for the **$M$th-order Krylov subspace**:

$$
\mathcal{K}_M(\mathcal{H}, \psi) = \mathrm{span}\{\psi,\, \mathcal{H}\psi,\, \mathcal{H}^2\psi,\, \ldots,\, \mathcal{H}^{M-1}\psi\}
$$

Starting from $v_1 = \psi/\|\psi\|$, successive Lanczos vectors are generated by applying $\mathcal{H}$ and orthogonalising via Gram–Schmidt:

$$
V_{k+1} = \mathcal{H}\,v_k - \sum_{j=1}^k H^L_{jk}\,v_j, \qquad
v_{k+1} = V_{k+1}/\|V_{k+1}\|, \qquad
H^L_{jk} = \langle v_j|\mathcal{H}|v_k\rangle
$$

Because $\mathcal{H}$ is Hermitian, $H^L$ is a **Hermitian tridiagonal matrix**. It is diagonalised as $H^L = U E^L U^\dagger$, and the wavefunction is propagated within the Krylov subspace:

$$
\psi(t+\Delta t) \approx V_M\cdot U\,e^{-iE^L\Delta t}\,U^\dagger\,e_1
$$

where $e_1 = (1, 0, \ldots, 0)^T$ is the representation of $\psi$ in the Lanczos basis.

**Key properties:**
- Larger $M$ allows much larger time steps.
- Near-unitary (to floating-point precision) for Hermitian $\mathcal{H}$.
- Each step requires $M$ sparse matrix–vector products.

---

## 7. Split-Operator Technique

The split-operator method exploits the additive structure $\mathcal{H} = \hat{T} + \hat{V}$ and the **Suzuki–Trotter expansion**. The kinetic-referenced second-order scheme is:

$$
e^{-i(\hat{T}+\hat{V})\Delta t}
\approx
e^{-i\hat{V}\Delta t/2}\;
e^{-i\hat{T}\Delta t}\;
e^{-i\hat{V}\Delta t/2}
+ \mathcal{O}(\Delta t^3)
$$

Each sub-operation is exact because:
- $\hat{V}$ is **diagonal in real space** → element-wise multiplication.
- $\hat{T}$ is **diagonal in Fourier (k) space** → apply via FFT:

$$
e^{-i\hat{T}\Delta t}\,\psi = \mathcal{F}^{-1}\!\left[e^{-ik^2\Delta t/2}\cdot\mathcal{F}[\psi]\right]
$$

Each time step therefore requires exactly **two FFTs and two IFFTs**.

**Key properties:**
- Always unitary and **unconditionally stable** (no constraint on $\Delta t$ from the kinetic energy).
- Easily extended to 2-D and 3-D.
- Implicitly uses **periodic boundary conditions** (from FFT), unlike the zero-BC methods above.

---

## 8. Choosing a Time Step

A practical estimate for the maximum safe time step comes from the kinetic energy of the fastest plane-wave component in the initial Gaussian wavepacket (with momentum spread $\sigma_k = 1/\sigma_x$):

$$
\Delta t \lesssim \frac{1}{E_k} = \frac{2}{(|k_0| + \sigma_k)^2}
$$

For example, with $k_0 = 0$ and $\sigma_x = 0.4$ a.u., this gives $\Delta t \approx 0.008$ a.u., consistent with numerical tests.  
For $k_0 = 10$ and $\sigma_x = 0.4$ a.u., the estimate drops to $\Delta t \approx 0.003$ a.u.

The RK4 method requires a factor of ~5–10 smaller step than CN for equivalent accuracy; Lanczos (with $M = 15$) can use steps ~4–10× larger than CN.

---

## 9. Boundary Conditions

| Method | Boundary Condition | Physical Interpretation |
|---|---|---|
| Crank–Nicolson | Zero (Dirichlet) | Infinite potential walls at boundaries |
| Runge–Kutta | Zero (Dirichlet) | Same as CN |
| Chebyshev | Zero (Dirichlet) | Same as CN |
| Lanczos | Zero (Dirichlet) | Same as CN |
| Split-Operator | Periodic (from FFT) | The domain wraps — right edge connects to left |

The distinction matters once the wavepacket reaches the boundary. CN/RK4/Chebyshev/Lanczos will reflect it; Split-Operator will wrap it to the other side. For wavepackets well within the domain, both behave identically.

---

## 10. Method Comparison Summary

| Property | CN | RK4 | Chebyshev | Lanczos | Split-Operator |
|---|---|---|---|---|---|
| Unitarity / norm conservation | Exact | Approximate | Near-exact | Near-exact | Exact |
| Time-step constraint | Moderate | Strict | Moderate | Large | None |
| Accuracy order | 2nd | 4th | Spectral | Spectral | 2nd |
| Cost per step | Sparse solve | Matrix–vector × 4 | Matrix–vector × N | Matrix–vector × M | 2 FFTs |
| Boundary condition | Zero | Zero | Zero | Zero | Periodic |
| Best suited for | General 1-D | Benchmarking | High accuracy | Large Δt | 2-D / 3-D, periodic |

---

## 11. References

1. Hartree atomic units — [Wikipedia](https://en.wikipedia.org/wiki/Hartree_atomic_units)  
2. Crank–Nicolson method — [Wikipedia](https://en.wikipedia.org/wiki/Crank%E2%80%93Nicolson_method)  
3. Finite difference (second-order central) — [Wikipedia](https://en.wikipedia.org/wiki/Finite_difference)  
4. Gaussian wave packet — [Wikipedia](https://en.wikipedia.org/wiki/Wave_packet#Gaussian_wave_packets_in_quantum_mechanics)  
5. QMsolve — [github.com/quantum-visualizations/qmsolve](https://github.com/quantum-visualizations/qmsolve)  
6. Runge–Kutta methods — [Wikipedia](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods)  
7. Chebyshev polynomials — [Wikipedia](https://en.wikipedia.org/wiki/Chebyshev_polynomials)  
8. `scipy.special.jv` — [SciPy docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.jv.html)  
9. Krylov subspace — [Wikipedia](https://en.wikipedia.org/wiki/Krylov_subspace)  
10. Tal-Ezer & Kosloff, *J. Chem. Phys.* **128**, 164116 (2008) — [DOI](https://aip.scitation.org/doi/full/10.1063/1.2916581)
