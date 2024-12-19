#pragma once

#include <functional>
#include <ostream>

#include "helpers.hpp"

namespace sip {

enum class Status {
  SOLVED = 0,
  ITERATION_LIMIT = 1,
  LINE_SEARCH_FAILURE = 2,
};

struct Settings {
  // The maximum number of iterations the solver can do.
  double max_iterations = 100;
  // The maximum allowed violation of the KKT system.
  double max_kkt_violation = 1e-6;
  // A parameter of the fraction-to-the-boundary rule.
  double tau_min = 0.995;
  // A parameter of the merit function and descent direction computation.
  double mu_min = 1e-12;
  // A regularization parameter, applied to the yy-block of the LHS of the
  // Newton-KKT system.
  double gamma_y = 1e-6;
  // A regularization parameter, applied to the zz-block of the LHS of the
  // Newton-KKT system.
  double gamma_z = 1e-6;
  // Determines when we accept a line search step, by the merit decrease and
  // slope.
  double armijo_factor = 1e-4;
  // Determines how much to backtrack at each line search iteration.
  double line_search_factor = 0.5;
  // Determines when we declare a line search failure.
  double line_search_min_step_size = 1e-6;
  // Whether to enable the usage of elastic variables.
  bool enable_elastics = false;
  // Determines how elastic variables are penalized in the cost function.
  double elastic_var_cost_coeff = 0.0;
  // When true, halts the optimization process if a good step is not found.
  bool enable_line_search_failures = false;
  // Determines whether we should print the solver logs.
  bool print_logs = true;
};

// For nicer googletest outputs.
auto operator<<(std::ostream &os, Status const &status) -> std::ostream &;

struct ModelCallbackInput {
  double *x;
};

struct ModelCallbackOutput {
  // NOTE: all sparse matrices should be represented in CSC format.

  // The objective and its first two derivatives.
  // NOTE: only the upper triangle should be filled in upper_hessian_f.
  // NOTE: upper_hessian_f should be a positive definite approximation.
  double f;
  double *gradient_f;
  SparseMatrix upper_hessian_f;

  // The equality constraints and their first derivative.
  double *c;
  SparseMatrix jacobian_c;

  // The inequality constraints and their first derivative.
  double *g;
  SparseMatrix jacobian_g;

  // To dynamically allocate the required memory.
  void reserve(int x_dim, int s_dim, int y_dim, int upper_hessian_f_nnz,
               int jacobian_c_nnz, int jacobian_g_nnz,
               bool is_jacobian_c_transposed, bool is_jacobian_g_transposed);
  void free();

  // For using pre-allocated (possibly statically allocated) memory.
  auto mem_assign(int x_dim, int s_dim, int y_dim, int upper_hessian_f_nnz,
                  int jacobian_c_nnz, int jacobian_g_nnz,
                  bool is_jacobian_c_transposed, bool is_jacobian_g_transposed,
                  unsigned char *mem_ptr) -> int;
};

struct Input {
  // NOTE: the user should ensure that no dynamic memory allocation happens
  //       when passing in model_callback and lin_sys_solver, possibly by
  //       wrapping them with std::cref (in case they are lambdas).

  using ModelCallback =
      std::function<void(const ModelCallbackInput &, ModelCallbackOutput **)>;

  // Callback type for solving Au + b = 0, where:
  // 1. u = (dx, ds, dy, dz);
  // 2. A = [ H + r1 I_x       0        C.T            G.T      ]
  //        [     0        S^{-1} Z      0             I_s      ]
  //        [     C            0      -r2 I_y           0       ]
  //        [     G           I_s        0      -(r3 + 1/p) I_z ];
  // 3. (H + r1 I_x) is symmetric and positive definite;
  // 4. np.triu(H) = upper_hessian_f;
  // 5. C is jacobian_c;
  // 6. G is jacobian_g;
  // 7. S = np.diag(s);
  // 8. Z = np.diag(z);
  // 9. r1, r2, r3 are non-negative regularization parameters;
  // 10. p is the penalty term on the elastic variables
  //     (inf when elastics are inactive);
  // 11. b = [     grad_L    ]
  //         [   z - mu / s  ]
  //         [       c       ]
  //         [ g + s - z / p ];
  // 12. grad_L = gradient_f + C.T y + G.T z;
  // 13. de = (-dz - (pe + z)) / p (when elastics are active);
  // 14. v = (dx, ds, dy, dz, de).

  // TODO(joao): make SIP build the 5x5 Newton-KKT system,
  //             and add a method in SLACG to solve that via
  //             the 3x3 solve and elimination of ds and de.

  // 3x3 upper_lhs:
  // [ upper_hessian_f    jacobian_c_t          jacobian_g_t    ]
  // [        0          -gamma_y * I_y               0         ]
  // [        0                0                 -sigma^{-1}    ]
  // Above, sigma = np.diag(z / (s + (gamma_z + 1/p) * z)),
  // ie W = S/Z and r3 = gamma_z + 1/p.

  // 3x3 nrhs:
  // [ gradient_f + jacobian_c_t @ y + jacobian_g_t @ z ]
  // [                          c                       ]
  // [               g(x) + mu / z - z / p              ]

  // 4x4 upper_lhs:
  // Builds the following matrix in CSC format:
  // [ upper_hessian_f       0        jacobian_c_t          jacobian_g_t    ]
  // [        0          S^{-1} Z          0                    I_s         ]
  // [        0              0      -gamma_y * I_y               0          ]
  // [        0              0             0         -(gamma_z + 1/p) * I_z ]

  // 4x4 rhs:
  // [ gradient_f + jacobian_c_t @ y + jacobian_g_t @ z ]
  // [                     z - mu / s                   ]
  // [                          c                       ]
  // [                   g + s - z / p                  ]

  // 5x5 upper_lhs:
  // [ upper_hessian_f       0        jacobian_c_t   jacobian_g_t     0  ]
  // [        0          S^{-1} Z          0              I_s         0  ]
  // [        0              0      -gamma_y * I_y         0          0  ]
  // [        0              0             0         -gamma_z I_z     I  ]
  // [        0              0             0               0         p I ]

  // 5x5 nrhs:
  // [ gradient_f + jacobian_c_t @ y + jacobian_g_t @ z ]
  // [                     z - mu / s                   ]
  // [                          c                       ]
  // [                      g + s + e                   ]
  // [                       pe + z                     ]

  using LinearSystemSolver = std::function<void(
      const ModelCallbackOutput &mco, const double *s, const double *y,
      const double *z, const double *e, const double mu, const double p,
      const double r1, const double r2, const double r3, const double new_lhs,
      double *dx, double *ds, double *dy, double *dz, double *de,
      double &kkt_error, double &lin_sys_error)>;

  // Callback for filling the ModelCallbackOutput object.
  ModelCallback model_callback;
  // An solver provided by the user to solve the Newton-KKT linear systems.
  LinearSystemSolver lin_sys_solver;
};

struct Output {
  // The exit status of the optimization process.
  Status exit_status;
  int num_iterations;
};

struct VariablesWorkspace {
  // The primal variables.
  double *x;
  // The slack variables.
  double *s;
  // The dual variables associated with the equality constraints.
  double *y;
  // The dual variables associated with the inequality constraints.
  double *z;
  // The elastic variables.
  double *e;
  // The next primal variables.
  double *next_x;
  // The next slack variables.
  double *next_s;
  // The next elastic variables.
  double *next_e;
  // The change to the primal variables.
  double *dx;
  // The change to the slack variables.
  double *ds;
  // The change to the dual variables associated with the equality constraints.
  double *dy;
  // The change to the dual variables associated with the inequality constraints.
  double *dz;
  // The change to the elastic variables.
  double *de;

  // To dynamically allocate the required memory.
  void reserve(int x_dim, int s_dim, int y_dim);
  void free();

  // For using pre-allocated (possibly statically allocated) memory.
  auto mem_assign(int x_dim, int s_dim, int y_dim, unsigned char *mem_ptr)
      -> int;
};

struct MiscellaneousWorkspace {
  // Stores g(x) + s.
  double *g_plus_s;
  // Stores g(x) + s (+ e, when applicable).
  double *g_plus_s_plus_e;

  // To dynamically allocate the required memory.
  void reserve(int s_dim);
  void free();

  // For using pre-allocated (possibly statically allocated) memory.
  auto mem_assign(int s_dim, unsigned char *mem_ptr) -> int;
};

// This data structure is used to avoid doing dynamic memory allocation inside
// of the solver, as well as avoiding excessive templating in the solver code.
struct Workspace {
  // The variable storage (for both primal and dual variables).
  // NOTE: vars will be consumed as a warm-start, and also to report the final
  // solution;
  //       other members need to be allocated, but not filled, as they are
  //       internal to the solver.
  VariablesWorkspace vars;
  // The model callback workspace.
  // NOTE: the ModelCallbackOutput object is owned by the user.
  ModelCallbackOutput *model_callback_output;
  // Stores miscellaneous items.
  MiscellaneousWorkspace miscellaneous_workspace;

  // To dynamically allocate the required memory.
  void reserve(int x_dim, int s_dim, int y_dim);
  void free();

  // For using pre-allocated (possibly statically allocated) memory.
  auto mem_assign(int x_dim, int s_dim, int y_dim, unsigned char *mem_ptr)
      -> int;
};

auto solve(const Input &input, const Settings &settings, Workspace &workspace,
           Output &output) -> void;

} // namespace sip
