# sip
This repository implements a sparse interior point solver in C++.

Our method optimizes nonlinear nonconvex optimization
problems of the form

$$\min\limits_{x} f(x) \qquad \mbox{s.t.} \quad c(x) = 0 \wedge g(x) <= 0$$.

The functions $f, c, g$ are required to be continuously differentiable,
to ensure that our line searches succeed.

We use [QDLDL](https://github.com/osqp/qdldl) to solve the
Newton-KKT linear systems at each step.

No dynamic memory allocation is done inside of the solver.
