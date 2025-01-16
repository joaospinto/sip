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

No dynamic memory allocation is done inside of the solver.


## Basic approach

We start by defining the Barrier-Lagrangian

$$
\mathcal{L}(x, s, y, z; \mu) =
f(x) - \mu \sum \limits_{i} \log(s_i) + y^T c(x) + z^T (g(x) + s).
$$

The first order optimality conditions, given by $\nabla \mathcal{L} = 0$, are

$$
\begin{align*} 
\nabla_x f(x) + J(c)(x)^T y + J(g)(x)^T &= 0 \\
s \circ z - \mu \mathbb{1} &= 0 \\
c(x) &= 0 \\
g(x) + s &= 0,
\end{align*}
$$

where $\circ$ represents elementwise multiplication and $\mathbb{1}$ represents
an all- $1$ vector.

Defining the KKT residual function

$$
k(x, s, y, z) = (\nabla_x f(x) + J(c)(x)^T y + J(g)(x)^T,
                 s \circ z - \mu \mathbb{1},
                 c(x),
                 g(x) + s),
$$

we can use Newton's method to find a zero of $k$.

Specifically, starting from a candidate point $(x_0, s_0, y_0, z_0)$,
we iteratively compute Newton steps defined by

$$
J(k)(x, s, y, z) (\Delta x, \Delta s, \Delta y, \Delta z) = -k(x, s, y, z).
$$

This yields:

$$ \begin{bmatrix}
   \nabla_{xx} \mathcal{L} & 0 & C.T & G.T \\
   0 & S^{-1} Z & 0 & I \\
   C & 0 & 0 & 0 \\
   G & I & 0 & 0
   \end{bmatrix}
\begin{bmatrix}
\Delta x \\
\Delta s \\
\Delta y \\
\Delta z
\end{bmatrix}
= - \begin{bmatrix}
    \nabla_x \mathcal{L} \\
    z - \mu S^{-1} \mathbb{1} \\
    c(x) \\
    g(x) + s
    \end{bmatrix}. $$

Next, we define the Augmented Barrier-Lagrangian

$$
\mathcal{A}(x, s, y, z; \mu, \eta) = \mathcal{L}(x, s, y, z; \mu) +
\frac{\eta}{2} (\lVert c(x) \rVert^2 + \lVert g(x) + s \rVert^2).
$$

Using $D( \cdot ; \cdot )$ to represent the directional derivative operator, note that

$$
\begin{align*} 
\frac{1}{2} D(\lVert c(x) \rVert^2; \Delta x) &=
(J(c)(x)^T c(x))^T \Delta x = c(x)^T (J(c)(x) \Delta x) \\
&= c(x)^T (-c(x)) = -\lVert c(x) \rVert^2
\end{align*} 
$$

and that

$$
\begin{align*} 
\frac{1}{2} D(\lVert (g(x) + s) \rVert^2; (\Delta x; \Delta s)) &=
(g(x) + s)^T (J(g)(x) \Delta x + \Delta s) \\
&= (g(x) + s)^T (-g(x) + s) = -\lVert g(x) + s \rVert^2
\end{align*} 
$$

Therefore,

$$
\begin{align*} 
D(\mathcal{A}(x, s, y, z; \mu, \eta); (\Delta x, \Delta s, \Delta y, \Delta z)) =
&D(\mathcal{L}(x, s, y, z; \mu); (\Delta x, \Delta s, \Delta y, \Delta z)) +
\frac{\eta}{2} D(\lVert c(x) \rVert^2 + \lVert g(x) + s \rVert^2; (\Delta x; \Delta s)) = \\
&-\begin{bmatrix}
  \Delta x & \Delta s & \Delta y & \Delta z
  \end{bmatrix}
\begin{bmatrix}
\nabla_{xx} \mathcal{L} & 0 & C.T & G.T \\
0 & S^{-1} Z & 0 & I \\
C & 0 & 0 & 0 \\
G & I & 0 & 0
\end{bmatrix}
\begin{bmatrix}
\Delta x \\
\Delta s \\
\Delta y \\
\Delta z
\end{bmatrix} \\
&- \eta ( \lVert c(x) \rVert^2 + \lVert g(x) + s \rVert^2 ).
\end{align*}
$$

In particular,

$$
\begin{align*} 
D(\mathcal{A}(x, s, y, z; \mu, \eta); (\Delta x, \Delta s, 0, 0)) =
&-\begin{bmatrix}
  \Delta x & \Delta s
  \end{bmatrix}
\begin{bmatrix}
\nabla_{xx} \mathcal{L} & 0 \\
0 & S^{-1} Z
\end{bmatrix}
\begin{bmatrix}
\Delta x \\
\Delta s
\end{bmatrix} \\
&- \eta ( \lVert c(x) \rVert^2 + \lVert g(x) + s \rVert^2 ) < 0,
\end{align*}
$$

as $s, z > 0$ and assuming $\nabla_{xx} \mathcal{L}$ is replaced with any
symmetric positive definite approximation.

This means that $(\Delta x, \Delta s, 0, 0)$ is always a descent direction
of $\mathcal{A}(x, s, y, z; \mu, \eta)$. Thus, the Augmented Barrier-Lagrangian
$\mathcal{A}(x, s, y, z; \mu, \eta)$ can be used as the merit function for a
line search over the primal variables $(x, s)$.

## Adding regularization

To do.
