# SIP

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

No dynamic memory allocation is done inside of the solver, as long as logging is off.


## Core Algorithm

SIP implements the Augmented Barrier-Lagrangian method, which can be seen as
a combination of the Primal Infeasible Interior Point and
the Augmented Lagrangian methods (as shown below).

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

### Linear system reformulation

Letting $\hat{\lambda} = \tilde{\lambda} + \Delta y, \hat{\nu} = \tilde{\nu} + \Delta z$,
we define $\Delta y \prime = \hat{\lambda} - \lambda$ and
$\Delta z \prime = \hat{\nu} - \nu$. Then

$$
\begin{align*}
\Delta y \prime &= \hat{\lambda} - \lambda
= (\hat{\lambda} - \tilde{\lambda}) + (\tilde{\lambda} - \lambda)
= \Delta y + \eta c(\xi) \\
\Delta z \prime &= \hat{\nu} - \nu
= (\hat{\nu} - \tilde{\nu}) + (\tilde{\nu} - \nu)
= \Delta z + \eta (g(\xi) + \sigma + \epsilon) .
\end{align*}
$$

We can now reformulate the linear system above in terms of $(\Delta y \prime, \Delta z \prime)$
instead of $(\Delta y, \Delta z)$:

$$
\begin{align*}
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
\Delta y \prime \\
\Delta z \prime
\end{bmatrix}
= - \begin{bmatrix}
    \nabla_x \mathcal{f}(\xi) + J(c)(\xi)^T \lambda + J(g)(\xi)^T \nu \\
    \nu - \mu \Sigma^{-1} \mathbb{1} \\
    \rho e + \nu \\
    c(\xi) \\
    g(\xi) + \sigma + \epsilon
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
\Delta y \prime \\
\Delta z \prime
\end{bmatrix}
= - \nabla \mathcal{L}(\xi, \sigma, \epsilon, \lambda, \nu)
\end{align*}
$$

This formulation has the advantage of being more numerically stable, due to having fewer dependencies on $\eta$.
It also allows us to directly access the non-augmented Newton-KKT residual, making termination checking simpler.
However, we still need to compute
$D(\mathcal{A}(x, s, e; \lambda, \nu, \mu, \eta); (\Delta x, \Delta s, \Delta e)) \mid_{(\xi, \sigma, \epsilon)}$.
This can be done efficiently (i.e. without incurring extra matrix-vector products) by noting that

$$
\begin{align*}
& D(\mathcal{A}(x, s, e; \lambda, \nu, \mu, \eta); (\Delta x, \Delta s, \Delta e)) \mid_{(\xi, \sigma, \epsilon)}
= \begin{bmatrix}
\Delta x & \Delta s & \Delta e
\end{bmatrix}
\begin{bmatrix}
\nabla_x \mathcal{A}(\xi, \sigma, \epsilon, \lambda, \nu, \mu) \\
\nabla_s \mathcal{A}(\xi, \sigma, \epsilon, \lambda, \nu, \mu) \\
\nabla_e \mathcal{A}(\xi, \sigma, \epsilon, \lambda, \nu, \mu)
\end{bmatrix} \\
= & \begin{bmatrix}
\Delta x & \Delta s & \Delta e
\end{bmatrix}
\left(
\begin{bmatrix}
\nabla_x \mathcal{L}(\xi, \sigma, \epsilon, \lambda, \nu, \mu) \\
\nabla_s \mathcal{L}(\xi, \sigma, \epsilon, \lambda, \nu, \mu) \\
\nabla_e \mathcal{L}(\xi, \sigma, \epsilon, \lambda, \nu, \mu)
\end{bmatrix} +
\eta
\begin{bmatrix}
J(c)(\xi)^T c(\xi) + J(g)(\xi) (g(\xi) + \sigma + \epsilon) \\
g(\xi) + \sigma + \epsilon \\
g(\xi) + \sigma + \epsilon
\end{bmatrix}
\right) \\
= & \begin{bmatrix}
\Delta x & \Delta s & \Delta e
\end{bmatrix}
\left(
\begin{bmatrix}
\nabla_x \mathcal{L}(\xi, \sigma, \epsilon, \lambda, \nu, \mu) \\
\nabla_s \mathcal{L}(\xi, \sigma, \epsilon, \lambda, \nu, \mu) \\
\nabla_e \mathcal{L}(\xi, \sigma, \epsilon, \lambda, \nu, \mu)
\end{bmatrix} +
\eta
\begin{bmatrix}
0 \\
g(\xi) + \sigma + \epsilon \\
g(\xi) + \sigma + \epsilon
\end{bmatrix}
\right) +
\eta c(\xi)^T J(c)(\xi) \Delta x +
\eta (g(\xi) + \sigma + \epsilon)^T J(g)(\xi) \Delta x \\
= & \begin{bmatrix}
\Delta x & \Delta s & \Delta e
\end{bmatrix}
\left(
\begin{bmatrix}
\nabla_x \mathcal{L}(\xi, \sigma, \epsilon, \lambda, \nu, \mu) \\
\nabla_s \mathcal{L}(\xi, \sigma, \epsilon, \lambda, \nu, \mu) \\
\nabla_e \mathcal{L}(\xi, \sigma, \epsilon, \lambda, \nu, \mu)
\end{bmatrix} +
\eta
\begin{bmatrix}
0 \\
g(\xi) + \sigma + \epsilon \\
g(\xi) + \sigma + \epsilon
\end{bmatrix}
\right) +
\eta c(\xi)^T \left( \frac{1}{\eta} \Delta y  - c(\xi) \right) +
\eta \left( g(\xi) + \sigma + \epsilon \right)^T \left( \frac{1}{\eta} \Delta z -
(g(\xi) + \sigma + \epsilon) - \Delta s - \Delta e \right) \\
= & \begin{bmatrix}
\Delta x & \Delta s & \Delta e
\end{bmatrix}
\begin{bmatrix}
\nabla_x \mathcal{L}(\xi, \sigma, \epsilon, \lambda, \nu, \mu) \\
\nabla_s \mathcal{L}(\xi, \sigma, \epsilon, \lambda, \nu, \mu) \\
\nabla_e \mathcal{L}(\xi, \sigma, \epsilon, \lambda, \nu, \mu)
\end{bmatrix} +
c(\xi)^T \Delta y +
(g(\xi) + \sigma + \epsilon)^T \Delta z -
\eta \left( \lVert c(\xi) \rVert^2 + \lVert g(\xi) + \sigma + \epsilon \rVert^2 \right) .
\end{align*}
$$

### Merit function and line search

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

Once a primal line search step size $\alpha$ is found (by checking the Armijo condition
while backtracking from $\alpha = 1$), the primal candidate solution is updated via

$$
\begin{align*}
\xi &\leftarrow \xi + \alpha \Delta x \\
\sigma &\leftarrow \max(\sigma + \alpha \Delta s, (1 - \tau) \sigma) \\
\epsilon &\leftarrow \epsilon + \alpha \Delta e \\
\end{align*}
$$

where $\tau$ is the fraction-to-the-boundary parameter, which is applied to
prevent $\sigma, \nu$ from approaching $0$ too quickly.

### Dual variable updates

The dual variable updates are scaled by $\beta_y, \beta_z$ to ensure that the merit function
decrease provided by the primal variable updates is not regressed beyond a constant
multiplicative factor.

Note that the only terms of $\\mathcal{A}(\xi, \sigma, \epsilon, \lambda, \nu)$ that depend
on $\lambda, \nu$  are $\lambda^T c(\xi)$ and $\nu^T (g(\xi) + \sigma + \epsilon)$.

In particular, when these inner products are negative, $\beta_y = \beta_z = 1$.

A fraction-to-the-boundary rule is applied to prevent $\nu$ from approaching $0$ too quickly.

$$
\begin{align*}
\lambda &\leftarrow \lambda + \beta_y \Delta y \prime \\
\nu &\leftarrow \max(\nu + \beta_z \Delta z \prime, (1 - \tau) \nu) ,
\end{align*}
$$

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
U &= (\nabla^2_{xx} \mathcal{L} + \eta J(c)^T J(c) + \eta J(g)^T V J(g))^{-1} ,
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
