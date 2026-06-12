# SIP

SIP is a sparse interior point solver for nonlinear optimization problems of
the form

$$
\begin{aligned}
\min_x \quad & f(x) \\
\textrm{s.t.}\quad & c(x) = 0, \\
& g(x) \leq 0.
\end{aligned}
$$

Internally, SIP introduces positive slack variables and solves the equivalent
system

$$
c(x) = 0, \qquad g(x) + s = 0, \qquad s > 0.
$$

The functions $f$, $c$, and $g$ are required to be continuously differentiable.
The user provides callbacks for model evaluation, matrix-vector products, and a
linear solver for the Newton-KKT systems.

## Linear System Callback

The factorization callback receives the diagonal quantities used by the reduced
Newton-KKT system

$$
\begin{bmatrix}
H + r_1 I & C^T & G^T \\
C & -\operatorname{diag}(r_2) & 0 \\
G & 0 & -\operatorname{diag}(r_3)
\end{bmatrix},
$$

where $H$ is the Lagrangian Hessian, $C = J(c)$, and $G = J(g)$. The callback
should factor this system and return whether the factorization succeeded and
the matrix has the desired inertia. SIP may increase the x-regularization
parameter and retry the factorization when the callback reports failure.

The solve callback then solves the factored system for the right-hand sides
assembled by SIP.

## Merit Function

SIP uses an augmented barrier-Lagrangian merit function based on

$$
f(x) - \mu \sum_i \log(s_i) + y^T c(x) + z^T (g(x) + s)
+ \frac{1}{2}\left(\|c(x)\|_{\eta_y}^2 + \|g(x) + s\|_{\eta_z}^2\right).
$$

The penalty parameters for equality and inequality constraints are stored
separately and updated per constraint.

## Memory

No dynamic memory allocation is done inside the solver once the `Workspace` has
been reserved or assigned preallocated memory, as long as diagnostic logging and
derivative checks are disabled.

## Examples

Examples and solver integrations are maintained in:

- [SIP Examples](https://github.com/joaospinto/sip_examples)
- [SIP QDLDL](https://github.com/joaospinto/sip_qdldl)
