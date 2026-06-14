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

## Examples

Examples and solver integrations are maintained in:

- [SIP Examples](https://github.com/joaospinto/sip_examples)
