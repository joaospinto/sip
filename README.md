# SIP
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
