# SIP

## Important note

SIP is still in active development, and not yet ready for external usage.

## Introduction

This repository implements a sparse interior point solver in C++.

Our method optimizes nonlinear nonconvex optimization problems of the form

$$\min\limits_{x} f(x) + \frac{\rho}{2} \lVert e \rVert^2 \qquad \mbox{s.t.}
  \quad c(x) = 0 \wedge g(x) + s + e = 0 \wedge s \geq 0$$.

The functions $f, c, g$ are required to be continuously differentiable.

The user is required to provide a linear system solver for the Newton-KKT systems.
This can be done, for example via
[SLACG](https://github.com/joaospinto/slacg)
or
[SIP_QDLDL](https://github.com/joaospinto/sip_qdldl).

Some examples with code can be found in the
[SIP Examples](https://github.com/joaospinto/sip_examples) repository.

No dynamic memory allocation is done inside of the solver.


## Basic approach

SIP implements a combination of the Primal Infeasible Interior Point and
Augmented Lagrangian methods.

We represent the barrier parameter by $\mu$ and the Augmented Lagrangian
penalty parameter by $\eta$.

We start by defining the Barrier-Lagrangian

$$
\mathcal{L}(x, s, y, z; \mu) =
f(x) - \mu \sum \limits_{i} \log(s_i) + y^T c(x) + z^T (g(x) + s).
$$

Next, we define the Augmented Barrier-Lagrangian

$$
\mathcal{A}(x, s, y, z; \mu, \eta) = \mathcal{L}(x, s, y, z; \mu) +
\frac{\eta}{2} (\lVert c(x) \rVert^2 + \lVert g(x) + s \rVert^2).
$$

We will use $(\chi, \sigma, \lambda, \nu)$ to refer to
the current $(x, s, y, z)$ iterate.

As we wish to employ a primal method, we compute $(\Delta x, \Delta s)$
by applying Newton's method to $\mathcal{A}(x, s; \lambda, \nu, \mu, \eta)$.

Below, we use $S, Z$ to represent the diagonal matrices containing $s, z$
along the diagonal, respectively. Moreover, we use $\mathbb{1}$ to represent
the all- $1$ vector, and $\circ$ to represent elementwise vector multiplication.

Noting that

$$
\nabla_{x, s} \mathcal{A}(x, s; \lambda, \nu, \mu, \eta) =
\begin{bmatrix}
\nabla_x f(x) + J(c)(x)^T (\lambda + \eta c(x)) + J(g)(x)^T (\nu + \eta (g(x) + s)) \\
z - \mu S^{-1} 1
\end{bmatrix} ,
$$

we let

$$
k(x, s, y, z; \lambda, \nu, \mu, \eta) =
\begin{bmatrix}
\nabla_x f(x) + J(c)(x)^T y + J(g)(x)^T z \\
s \circ z - \mu 1 \\
c(x) + \frac{\lambda - y}{\eta} \\
g(x) + s + \frac{\nu - z}{\eta}
\end{bmatrix} ,
$$

and find $(\Delta x, \Delta s)$ by taking a Newton step on $k$.

Above, we introduced auxiliary variables $y, z$, meant to represent
$\lambda + \eta c(x), \nu + \eta (g(x) + s)$ respectively; note that these would
be the next Lagrange multiplier estimates in a pure Augmented Lagrangian setting.
This is done in order to prevent fill-in (via the presence of any
$J(c)(x)^T J(c)(x)$ or $J(g)(x)^T J(g)(x)$ terms) in the linear system we use
below to compute $(\Delta x, \Delta s)$.

Letting $\tilde{\lambda} = \lambda + \eta c(\chi), \tilde{\nu} = \nu + \eta (g(\chi) + \sigma)$,
we can take a Newton step via

$$
J(k)(\chi, \sigma, \tilde{\lambda}, \tilde{\nu})
\begin{bmatrix}
\Delta x \\
\Delta s \\
\Delta y \\
\Delta z
\end{bmatrix} =
-k(\chi, \sigma, \tilde{\lambda}, \tilde{\nu})
\Leftrightarrow
\begin{bmatrix}
H_x(\mathcal{A})(\chi, \sigma, \lambda, \nu, \mu, \eta) & 0 & J(c)(\chi)^T & J(g)(\chi)^T \\
0 & S^{-1} Z & 0 & I \\
J(c)(\chi) & 0 & -\frac{1}{\eta} I & 0 \\
J(g)(\chi) & I & 0 & -\frac{1}{\eta} I
\end{bmatrix}
\begin{bmatrix}
\Delta x \\
\Delta s \\
\Delta y \\
\Delta z
\end{bmatrix}
= - \begin{bmatrix}
    \nabla_x \mathcal{f}(\chi) + J(c)(\chi)^T \tilde{\lambda} + J(g)(\chi)^T \tilde{\nu} \\
    z - \mu S^{-1} \mathbb{1} \\
    0 \\
    0
    \end{bmatrix}. $$

Using $D( \cdot ; \cdot )$ to represent the directional derivative operator, note that

$$
\begin{align*} 
D(\mathcal{A}(x, s; \lambda, \nu, \mu, \eta); (\Delta x, \Delta s)) \mid_{(\chi, \sigma)} &=
\begin{bmatrix}
\Delta x & \Delta s
\end{bmatrix}
\begin{bmatrix}
\nabla_x \mathcal{f}(\chi) + J(c)(\chi)^T \tilde{\lambda} + J(g)(\chi)^T \tilde{\nu} \\
z - \mu S^{-1} \mathbb{1}
\end{bmatrix} \\
&= \begin{bmatrix}
   \Delta x & \Delta s & 0 & 0
   \end{bmatrix}
\begin{bmatrix}
   \nabla_x \mathcal{f}(\chi) + J(c)(\chi)^T \tilde{\lambda} + J(g)(\chi)^T \tilde{\nu} \\
   z - \mu S^{-1} \mathbb{1} \\
   0 \\
   0
   \end{bmatrix} \\
&= - \begin{bmatrix}
     \Delta x & \Delta s & 0 & 0
     \end{bmatrix}^T 
\begin{bmatrix}
H_x(\mathcal{A})(\chi, \sigma, \lambda, \nu, \mu, \eta) & 0 & J(c)(\chi)^T & J(g)(\chi)^T \\
0 & S^{-1} Z & 0 & I \\
J(c)(\chi) & 0 & -\frac{1}{\eta} I & 0 \\
J(g)(\chi) & I & 0 & -\frac{1}{\eta} I
\end{bmatrix}
\begin{bmatrix}
\Delta x \\
\Delta s \\
0 \\
0
\end{bmatrix} \\
&= - \begin{bmatrix}
     \Delta x & \Delta s
     \end{bmatrix}^T 
\begin{bmatrix}
H_x(\mathcal{A})(\chi, \sigma, \lambda, \nu, \mu, \eta) & 0 \\
0 & S^{-1} Z
\end{bmatrix}
\begin{bmatrix}
\Delta x \\
\Delta s
\end{bmatrix} < 0 ,
\end{align*}
$$

assuming that $H_x(\mathcal{A})(\chi, \sigma, \lambda, \nu, \mu, \eta)$ is replaced
with any symmetric positive definite approximation, as $s, z > 0$,
unless $(\Delta x, \Delta s) = (0, 0)$, in which case we have converged.

This means that $(\Delta x, \Delta s)$ is always a descent direction
of $\mathcal{A}(x, s; \lambda, \nu, \mu, \eta)$.
Thus, the Augmented Barrier-Lagrangian $\mathcal{A}(x, s; \lambda, \nu, \mu, \eta)$
can be used as the merit function for a line search over the primal variables $(x, s)$.

Once a primal line search step size $\alpha$ is found, the candidate solution
is updated via

$$
\begin{align*}
\chi &\leftarrow \chi + \alpha \Delta x \\
\sigma &\leftarrow \sigma + \alpha \Delta s \\
\lambda &\leftarrow \tilde{\lambda} + \alpha \Delta y \\
\nu &\leftarrow \tilde{\nu} + \alpha \Delta z .
\end{align*}
$$

Note that $\alpha$ is selected so that $\sigma$ remains positive; moreover, a
fraction-to-the-boundary rule is applied to prevent $\sigma, \nu$
from approaching $0$ too quickly.

### Solving the linear system

In this section, we show that 

$$ \begin{bmatrix}
   H_x(\mathcal{A}) & 0 & J(c)^T & J(g)^T \\
   0 & S^{-1} Z & 0 & I \\
   J(c) & 0 & -\frac{1}{\eta} I & 0 \\
   J(g) & I & 0 & -\frac{1}{\eta} I
   \end{bmatrix} , $$

is invertible, assuming again that $H_x(\mathcal{A})$ is replaced
with any symmetric positive definite approximation.

To prove this, letting

$$
\begin{align*} 
W &= Z^{-1} S \\
V &= (W + \frac{1}{\eta} I)^{-1} \\
U &= (H_x(\mathcal{A}) + \eta J(c)^T J(c) + \eta J(g)^T J(g))^{-1} ,
\end{align*} 
$$

note that

$$
\begin{align*} 
& \begin{bmatrix}
  H_x(\mathcal{A}) & 0 & J(c)^T & J(g)^T \\
  0 & S^{-1} Z & 0 & I \\
  J(c) & 0 & -\frac{1}{\eta} I & 0 \\
  J(g) & I & 0 & -\frac{1}{\eta} I
  \end{bmatrix}
  \begin{bmatrix}
  \Delta x \\
  \Delta s \\
  \Delta y \\
  \Delta z
  \end{bmatrix} =
  -\begin{bmatrix}
   r_x \\
   r_s \\
   r_y \\
   r_z
   \end{bmatrix} \\
\Leftrightarrow
& \begin{bmatrix}
  H_x(\mathcal{A}) & J(c)^T & J(g)^T \\
  J(c) & -\frac{1}{\eta} I & 0 \\
  J(g) & 0 & -W -\frac{1}{\eta} I
  \end{bmatrix}
  \begin{bmatrix}
  \Delta x \\
  \Delta y \\
  \Delta z
  \end{bmatrix} =
  -\begin{bmatrix}
   r_x \\
   r_y \\
   r_z - W r_s
   \end{bmatrix} \\
\Leftrightarrow
& \begin{bmatrix}
  H_x(\mathcal{A}) + \eta J(c)^T J(c) & J(g)^T \\
  J(g) & -W -\frac{1}{\eta} I
  \end{bmatrix}
  \begin{bmatrix}
  \Delta x \\
  \Delta z
  \end{bmatrix} =
  -\begin{bmatrix}
   r_x + \eta J(c)^T r_y \\
   r_z - W r_s
   \end{bmatrix} \\
\Leftrightarrow
& (H_x(\mathcal{A}) + \eta J(c)^T J(c) + \eta J(g)^T J(g)) \Delta x =
   -(r_x + \eta J(c)^T r_y + J(g)^T V (r_z - W r_s)) \\
\Leftrightarrow
& \Delta x =
   -U (r_x + \eta J(c)^T r_y + J(g)^T V (r_z - W r_s))
\end{align*} 
$$

where we eliminated $\Delta s, \Delta y, \Delta z$ respectively via

$$
\begin{align*} 
\Delta s &= -Z^{-1} S (\Delta z + r_s), \\
\Delta y &= \eta (J(c) \Delta x + r_y), \\
\Delta z &= V (J(g) \Delta x + r_z - W r_s) .
\end{align*} 
$$

Depending on the sparsity pattern of the matrices involved,
doing this block-elimination (in full or in part) may be reasonable,
although typically it is faster to solve the block-3x3 formulation above
via a sparse $L D L^T$ decomposition.

<!---
TODO(joao):
1. add elastics to the docs.
2. implement everything.
-->
