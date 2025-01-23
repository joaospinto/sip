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
\mathcal{L}(x, s, e, y, z; \mu) =
f(x) + \frac{\rho}{2} \lVert e \rVert^2 - \mu \sum \limits_{i} \log(s_i) +
y^T c(x) + z^T (g(x) + s + e).
$$

Next, we define the Augmented Barrier-Lagrangian

$$
\mathcal{A}(x, s, e, y, z; \mu, \eta) = \mathcal{L}(x, s, e, y, z; \mu) +
\frac{\eta}{2} (\lVert c(x) \rVert^2 + \lVert g(x) + s + e\rVert^2).
$$

We will use $(\xi, \sigma, \epsilon, \lambda, \nu)$ to refer to
the current $(x, s, e, y, z)$ iterate. We use $\Sigma, \Pi$ to denote
the diagonal matrices with entries $\sigma, \nu$ respectively.

As we wish to employ a primal method, we compute $(\Delta x, \Delta s, \Delta e)$
by applying Newton's method to $\mathcal{A}(x, s, e; \lambda, \nu, \mu, \eta)$.

Below, we use $S, Z$ to represent the diagonal matrices containing $s, z$
along the diagonal, respectively. Moreover, we use $\mathbb{1}$ to represent
the all- $1$ vector, and $\circ$ to represent elementwise vector multiplication.

Noting that

$$
\nabla_{x, s, e} \mathcal{A}(x, s, e; \lambda, \nu, \mu, \eta) =
\begin{bmatrix}
\nabla_x f(x) + J(c)(x)^T (\lambda + \eta c(x)) + J(g)(x)^T (\nu + \eta (g(x) + s + e)) \\
\nu + \eta (g(x) + s + e) - \mu S^{-1} 1 \\
\rho e + \nu + \eta (g(x) + s + e)
\end{bmatrix} ,
$$

we let

$$
k(x, s, e, y, z; \lambda, \nu, \mu, \eta) =
\begin{bmatrix}
\nabla_x f(x) + J(c)(x)^T y + J(g)(x)^T z \\
s \circ z - \mu 1 \\
\rho e + z \\
c(x) + \frac{\lambda - y}{\eta} \\
g(x) + s + e + \frac{\nu - z}{\eta}
\end{bmatrix} ,
$$

and find $(\Delta x, \Delta s, \Delta e, \Delta y, \Delta z)$ by taking a Newton step on $k$.

Above, we introduced auxiliary variables $y, z$, meant to represent
$\lambda + \eta c(x), \nu + \eta (g(x) + s + e)$ respectively; note that these would
be the next Lagrange multiplier estimates in a pure Augmented Lagrangian setting,
modulo a $\max(\cdot, 0)$ projection in the case of $z$.
This is done in order to prevent fill-in (via the presence of any
$J(c)(x)^T J(c)(x)$ or $J(g)(x)^T J(g)(x)$ terms) in the linear system we use
below to compute $(\Delta x, \Delta s, \Delta e)$.

Letting $\tilde{\lambda} = \lambda + \eta c(\xi), \tilde{\nu} = \nu + \eta (g(\xi) + \sigma + \epsilon)$,
we can take a Newton step via

$$
\begin{align*}
& J(k)(\xi, \sigma, \epsilon, \tilde{\lambda}, \tilde{\nu})
\begin{bmatrix}
\Delta x \\
\Delta s \\
\Delta e \\
\Delta y \\
\Delta z
\end{bmatrix} =
-k(\xi, \sigma, \epsilon, \tilde{\lambda}, \tilde{\nu})
\Leftrightarrow \\
& \begin{bmatrix}
  \nabla^2_{xx} f( \xi ) + \tilde{\lambda}^T \nabla^2_{xx} c(\xi) + \tilde{\nu}^T \nabla^2_{xx} g (\xi) & 0 & 0 & J(c)(\xi)^T & J(g)(\xi)^T \\
  0 & \Sigma^{-1} \Pi & 0 & 0 & I \\
  0 & 0 & \rho I & 0 & I \\
  J(c)(\xi) & 0 & 0 & -\frac{1}{\eta} I & 0 \\
  J(g)(\xi) & I & I & 0 & -\frac{1}{\eta} I
  \end{bmatrix}
\begin{bmatrix}
\Delta x \\
\Delta s \\
\Delta e \\
\Delta y \\
\Delta z
\end{bmatrix}
= - \begin{bmatrix}
    \nabla_x \mathcal{f}(\xi) + J(c)(\xi)^T \tilde{\lambda} + J(g)(\xi)^T \tilde{\nu} \\
    \tilde{\nu} - \mu \Sigma^{-1} \mathbb{1} \\
    \rho e + \tilde{\nu} \\
    0 \\
    0
    \end{bmatrix} \Leftrightarrow \\
& \begin{bmatrix}
  \nabla^2_{xx} \mathcal{L}(\xi, \sigma, \tilde{\lambda}, \tilde{\nu}, \mu, \eta) & 0 & 0 & J(c)(\xi)^T & J(g)(\xi)^T \\
  0 & \Sigma^{-1} \Pi & 0 & 0 & I \\
  0 & 0 & \rho I & 0 & I \\
  J(c)(\xi) & 0 & 0 & -\frac{1}{\eta} I & 0 \\
  J(g)(\xi) & I & I & 0 & -\frac{1}{\eta} I
  \end{bmatrix}
\begin{bmatrix}
\Delta x \\
\Delta s \\
\Delta e \\
\Delta y \\
\Delta z
\end{bmatrix}
= - \begin{bmatrix}
    \nabla_x \mathcal{L}(\xi, \sigma, \tilde{\lambda}, \tilde{\nu}, \mu) \\
    \nabla_s \mathcal{L}(\xi, \sigma, \tilde{\lambda}, \tilde{\nu}, \mu) \\
    \nabla_e \mathcal{L}(\xi, \sigma, \tilde{\lambda}, \tilde{\nu}, \mu) \\
    0 \\
    0
    \end{bmatrix} \Leftrightarrow \\
& \begin{bmatrix}
  \nabla^2_{xx} \mathcal{L}(\xi, \sigma, \tilde{\lambda}, \tilde{\nu}, \mu, \eta) & 0 & 0 & J(c)(\xi)^T & J(g)(\xi)^T \\
  0 & \Sigma^{-1} \Pi & 0 & 0 & I \\
  0 & 0 & \rho I & 0 & I \\
  J(c)(\xi) & 0 & 0 & -\frac{1}{\eta} I & 0 \\
  J(g)(\xi) & I & I & 0 & -\frac{1}{\eta} I
  \end{bmatrix}
\begin{bmatrix}
\Delta x \\
\Delta s \\
\Delta e \\
\Delta y \\
\Delta z
\end{bmatrix}
= - \begin{bmatrix}
    \nabla_x \mathcal{A}(\xi, \sigma, \lambda, \nu, \mu) \\
    \nabla_s \mathcal{A}(\xi, \sigma, \lambda, \nu, \mu) \\
    \nabla_e \mathcal{A}(\xi, \sigma, \lambda, \nu, \mu) \\
    0 \\
    0
    \end{bmatrix} .
\end{align*}
$$

Note that

$$
\nabla^2_{xx} \mathcal{A}(\xi, \sigma, \lambda, \nu, \mu, \eta)
= \nabla^2_{xx} \mathcal{L}(\xi, \sigma, \tilde{\lambda}, \tilde{\nu}, \mu, \eta) +
\eta (J(c)(\xi)^T J(c)(\xi) + J(g)(\xi)^T J(g)(\xi)).
$$

Using $D( \cdot ; \cdot )$ to represent the directional derivative operator, note that

$$
\begin{align*} 
D(\mathcal{A}(x, s, e; \lambda, \nu, \mu, \eta); (\Delta x, \Delta s, \Delta e)) \mid_{(\xi, \sigma, \epsilon)} &=
\begin{bmatrix}
\Delta x & \Delta s & \Delta e
\end{bmatrix}
\begin{bmatrix}
\nabla_x \mathcal{A}(\xi, \sigma, \epsilon, \lambda, \nu, \mu) \\
\nabla_s \mathcal{A}(\xi, \sigma, \epsilon, \lambda, \nu, \mu) \\
\nabla_e \mathcal{A}(\xi, \sigma, \epsilon, \lambda, \nu, \mu)
\end{bmatrix} \\
&= \begin{bmatrix}
   \Delta x & \Delta s & \Delta e & 0 & 0
   \end{bmatrix}
\begin{bmatrix}
\nabla_x \mathcal{A}(\xi, \sigma, \epsilon, \lambda, \nu, \mu) \\
\nabla_s \mathcal{A}(\xi, \sigma, \epsilon, \lambda, \nu, \mu) \\
\nabla_e \mathcal{A}(\xi, \sigma, \epsilon, \lambda, \nu, \mu) \\
0 \\
0
\end{bmatrix} \\
&= - \begin{bmatrix}
     \Delta x & \Delta s & \Delta e & 0 & 0
     \end{bmatrix}
\begin{bmatrix}
\nabla^2_{xx} \mathcal{L}(\xi, \sigma, \tilde{\lambda}, \tilde{\nu}, \mu, \eta) & 0 & 0 & J(c)(\xi)^T & J(g)(\xi)^T \\
0 & \Sigma^{-1} \Pi & 0 & 0 & I \\
0 & 0 & \rho I & 0 & I \\
J(c)(\xi) & 0 & 0 & -\frac{1}{\eta} I & 0 \\
J(g)(\xi) & I & I & 0 & -\frac{1}{\eta} I
\end{bmatrix}
\begin{bmatrix}
\Delta x \\
\Delta s \\
\Delta e \\
\Delta y \\
\Delta z
\end{bmatrix} \\
&= - \begin{bmatrix}
     \Delta x & \Delta s & \Delta e
     \end{bmatrix} 
\begin{bmatrix}
\nabla^2_{xx} \mathcal{L}(\xi, \sigma, \tilde{\lambda}, \tilde{\nu}, \tilde{\mu}, \eta) & 0 & 0 \\
0 & \Sigma^{-1} \Pi \\
0 & 0 & \rho I
\end{bmatrix}
\begin{bmatrix}
\Delta x \\
\Delta s \\
\Delta e
\end{bmatrix} -
\begin{bmatrix}
\Delta x & \Delta s & \Delta e
\end{bmatrix}
\begin{bmatrix}
J(c)(\xi)^T & J(g)(\xi)^T \\
0 & I \\
0 & I
\end{bmatrix}
\begin{bmatrix}
\Delta y \\
\Delta z
\end{bmatrix} \\
&= - \begin{bmatrix}
     \Delta x & \Delta s & \Delta e
     \end{bmatrix} 
\begin{bmatrix}
\nabla^2_{xx} \mathcal{L}(\xi, \sigma, \tilde{\lambda}, \tilde{\nu}, \tilde{\mu}, \eta) & 0 & 0 \\
0 & \Sigma^{-1} \Pi \\
0 & 0 & \rho I
\end{bmatrix}
\begin{bmatrix}
\Delta x \\
\Delta s \\
\Delta e
\end{bmatrix} -
\begin{bmatrix}
J(c)(\xi) \Delta x \\
J(g)(\xi) \Delta x + \Delta s + \Delta e
\end{bmatrix}^T
\begin{bmatrix}
\eta (J(c)(\xi) \Delta x) \\
\eta (J(g)(\xi) \Delta x + \Delta s + \Delta e)
\end{bmatrix} \\
&= - \begin{bmatrix}
     \Delta x & \Delta s & \Delta e
     \end{bmatrix} 
\begin{bmatrix}
\nabla^2_{xx} \mathcal{L}(\xi, \sigma, \tilde{\lambda}, \tilde{\nu}, \tilde{\mu}, \eta) & 0 & 0 \\
0 & \Sigma^{-1} \Pi \\
0 & 0 & \rho I
\end{bmatrix}
\begin{bmatrix}
\Delta x \\
\Delta s \\
\Delta e
\end{bmatrix} -
\eta (\lVert J(c)(\xi) \Delta x \rVert^2 + \lVert J(g)(\xi) \Delta x + \Delta s + \Delta e \rVert^2) < 0,
\end{align*}
$$

assuming that $\nabla^2_{xx} \mathcal{L}(\xi, \sigma, \tilde{\lambda}, \tilde{\nu}, \mu, \eta)$
is replaced with any symmetric positive definite approximation, as $\sigma, \nu, \rho > 0$,
unless $(\Delta x, \Delta s, \Delta e) = (0, 0, 0)$, in which case we have converged.

This means that $(\Delta x, \Delta s, \Delta e)$ is always a descent direction
of $\mathcal{A}(x, s, e; \lambda, \nu, \mu, \eta)$.
Thus, the Augmented Barrier-Lagrangian $\mathcal{A}(x, s, e; \lambda, \nu, \mu, \eta)$
can be used as the merit function for a line search over the primal variables $(x, s, e)$.

Once a primal line search step size $\alpha$ is found, the candidate solution
is updated via

$$
\begin{align*}
\xi &\leftarrow \xi + \alpha \Delta x \\
\sigma &\leftarrow \max(\sigma + \alpha \Delta s, (1 - \tau) \sigma) \\
\epsilon &\leftarrow \epsilon + \alpha \Delta e \\
\lambda &\leftarrow \tilde{\lambda} + \alpha \Delta y \\
\nu &\leftarrow \max(\tilde{\nu} + \alpha \Delta z, (1 - \tau) \nu) ,
\end{align*}
$$

where $\tau$ is the fraction-to-the-boundary parameter, which is applied to
prevent $\sigma, \nu$ from approaching $0$ too quickly.

### Solving the linear system

In this section, we show that 

$$ \begin{bmatrix}
   \nabla^2_{xx} \mathcal{L} & 0 & 0 & J(c)^T & J(g)^T \\
   0 & \Sigma^{-1} \Pi & 0 & 0 & I \\
   0 & 0 & \rho I & 0 & I \\
   J(c) & 0 & 0 & -\frac{1}{\eta} I & 0 \\
   J(g) & I & I & 0 & -\frac{1}{\eta} I
   \end{bmatrix} , $$

is invertible, assuming again that $\nabla^2_{xx} \mathcal{L}$ is replaced
with any symmetric positive definite approximation.

To prove this, letting

$$
\begin{align*} 
W &= \Pi^{-1} \Sigma \\
V &= (W + (\frac{1}{\eta} + \frac{1}{\rho}) I)^{-1} \\
U &= (\nabla^2_{xx} \mathcal{L} + \eta J(c)^T J(c) + \eta J(g)^T J(g))^{-1} ,
\end{align*} 
$$

note that

$$
\begin{align*} 
& \begin{bmatrix}
  \nabla^2_{xx} \mathcal{L} & 0 & 0 & J(c)^T & J(g)^T \\
  0 & \Sigma^{-1} \Pi & 0 & 0 & I \\
   0 & 0 & \rho I & 0 & I \\
  J(c) & 0 & 0 & -\frac{1}{\eta} I & 0 \\
  J(g) & I & I & 0 & -\frac{1}{\eta} I
  \end{bmatrix}
  \begin{bmatrix}
  \Delta x \\
  \Delta s \\
  \Delta e \\
  \Delta y \\
  \Delta z
  \end{bmatrix} =
  -\begin{bmatrix}
   r_x \\
   r_s \\
   r_e \\
   r_y \\
   r_z
   \end{bmatrix} \\
\Leftrightarrow
& \begin{bmatrix}
  \nabla^2_{xx} \mathcal{L} & 0 & J(c)^T & J(g)^T \\
  0 & \Sigma^{-1} \Pi & 0 & I \\
  J(c) & 0 & -\frac{1}{\eta} I & 0 \\
  J(g) & I & 0 & -(\frac{1}{\eta} + \frac{1}{\rho}) I
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
   r_z - \frac{1}{\rho} r_e
   \end{bmatrix} \\
\Leftrightarrow
& \begin{bmatrix}
  \nabla^2_{xx} \mathcal{L} & J(c)^T & J(g)^T \\
  J(c) & -\frac{1}{\eta} I & 0 \\
  J(g) & 0 & -W -(\frac{1}{\eta} + \frac{1}{\rho}) I
  \end{bmatrix}
  \begin{bmatrix}
  \Delta x \\
  \Delta y \\
  \Delta z
  \end{bmatrix} =
  -\begin{bmatrix}
   r_x \\
   r_y \\
   r_z - \frac{1}{\rho} r_e - W r_s
   \end{bmatrix} \\
\Leftrightarrow
& \begin{bmatrix}
  \nabla^2_{xx} \mathcal{L} + \eta J(c)^T J(c) & J(g)^T \\
  J(g) & -W -(\frac{1}{\eta} + \frac{1}{\rho}) I
  \end{bmatrix}
  \begin{bmatrix}
  \Delta x \\
  \Delta z
  \end{bmatrix} =
  -\begin{bmatrix}
   r_x + \eta J(c)^T r_y \\
   r_z - \frac{1}{\rho} r_e - W r_s
   \end{bmatrix} \\
\Leftrightarrow
& (\nabla^2_{xx} \mathcal{L} + \eta J(c)^T J(c) + \eta J(g)^T V J(g)) \Delta x =
   -(r_x + \eta J(c)^T r_y + J(g)^T V (r_z - \frac{1}{\rho} r_e - W r_s)) \\
\Leftrightarrow
& \Delta x =
   -U (r_x + \eta J(c)^T r_y + J(g)^T V (r_z - \frac{1}{\rho} r_e - W r_s))
\end{align*} 
$$

where we eliminated $\Delta s, \Delta y, \Delta z$ respectively via

$$
\begin{align*}
\Delta e &= - \frac{1}{\rho}(\Delta z + r_e) \\
\Delta s &= -\Pi^{-1} \Sigma (\Delta z + r_s), \\
\Delta y &= \eta (J(c) \Delta x + r_y), \\
\Delta z &= V (J(g) \Delta x + r_z - \frac{1}{\rho} r_e - W r_s) .
\end{align*} 
$$

Depending on the sparsity pattern of the matrices involved,
doing this block-elimination (in full or in part) may be reasonable,
although typically it is faster to solve the block-3x3 formulation above
via a sparse $L D L^T$ decomposition.
