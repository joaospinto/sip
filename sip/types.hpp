#pragma once

#include <functional>
#include <ostream>

namespace sip {

enum class Status {
  SOLVED = 0,
  SUBOPTIMAL = 1,
  LOCALLY_INFEASIBLE = 2,
  ITERATION_LIMIT = 3,
  LINE_SEARCH_ITERATION_LIMIT = 4,
  LINE_SEARCH_FAILURE = 5,
  TIMEOUT = 6,
  FAILED_CHECK = 7,
};

// For nicer googletest outputs.
auto operator<<(std::ostream &os, Status const &status) -> std::ostream &;

struct Settings {
  // The maximum number of iterations the solver can do.
  int max_iterations = 100;
  // The maximum cumulative number of line search iterations.
  int max_ls_iterations = 500;
  // The number of iterative refinement steps.
  int num_iterative_refinement_steps = 1;
  // The maximum allowed violation of the KKT system.
  double max_kkt_violation = 1e-8;
  // The maximum allowed constraint violation to declare suboptimality.
  double max_suboptimal_constraint_violation = 1e-2;
  // The maximum allowed merit function slope.
  double max_merit_slope = 1e-16;
  // The initial x-regularizatino to be applied on the LHS.
  double initial_regularization = 1e-3;
  // The multiplicative decay of the x-regularization coefficient.
  double regularization_decay_factor = 0.5;
  // A parameter of the fraction-to-the-boundary rule.
  double tau = 0.995;
  // Determines whether we start with alpha=alpha_s_max or alpha=1.
  bool start_ls_with_alpha_s_max = false;
  // A parameter of the merit function and descent direction computation.
  double initial_mu = 1e-3;
  // Determines how much mu decreases per iteration.
  double mu_update_factor = 0.5;
  // The minimum barrier coefficient.
  double mu_min = 1e-16;
  // The initial penalty parameter of the Augmented Lagrangian.
  double initial_penalty_parameter = 1e3;
  // Minimum acceptable constraint violation ratio for eta to not increase.
  double min_acceptable_constraint_violation_ratio = 0.25;
  // By what factor to increase eta.
  double penalty_parameter_increase_factor = 10.0;
  // By what factor to decrease eta.
  double penalty_parameter_decrease_factor = 1.0;
  // The maximum allowed penalty parameter in the AL merit function.
  double max_penalty_parameter = 1e9;
  // Determines when we accept a line search step, by the merit decrease and
  // slope.
  double armijo_factor = 1e-4;
  // Determines how much to backtrack at each line search iteration.
  double line_search_factor = 0.5;
  // Determines when we declare a line search failure.
  double line_search_min_step_size = 1e-6;
  // When the merit slope becomes larger than this, no line search is done.
  double min_merit_slope_to_skip_line_search = -1e-3;
  // How much of the primal merit decrease to allow losing on the dual update.
  double dual_armijo_factor = 0.5;
  // Whether to enable the usage of elastic variables.
  bool enable_elastics = false;
  // Determines how elastic variables are penalized in the cost function.
  double elastic_var_cost_coeff = 0.0;
  // When true, halts the optimization process if a good step is not found.
  bool enable_line_search_failures = false;
  // Determines whether we should print the solver logs.
  bool print_logs = true;
  // Determines whether we should print the line search logs.
  bool print_line_search_logs = true;
  // Determines whether we should print the search direction computation logs.
  bool print_search_direction_logs = true;
  // Whether to print derivative check logs when something looks off.
  bool print_derivative_check_logs = true;
  // When running derivative checks, will only use the search direction when
  // true.
  bool only_check_search_direction_slope = false;
  // Handle checks with assert calls.
  bool assert_checks_pass = false;
};

struct ModelCallbackInput {
  // The primal variables.
  double *x;
  // The Lagrange multipliers associated with the equality constraints
  double *y;
  // The Lagrange multipliers associated with the inequality constraints
  double *z;
  // Whether x differs from the value from the last call.
  bool new_x;
  // Whether y differs from the value from the last call.
  bool new_y;
  // Whether z differs from the value from the last call.
  bool new_z;
};

struct Input {
  // NOTE: the user may ensure that no dynamic memory allocation happens
  //       when passing in the callbacks below, for example, by declaring
  //       them as lambdas and wrapping them with std::cref.

  // NOTE: the factor/solve callbacks should solve Kv = b, where:
  // 1. K = [ H + r1 I_x      C.T        G.T   ]
  //        [     C        -r2 * I_y      0    ]
  //        [     G            0       -r3 I_z ]
  // 2. (H + r1 I_x) is symmetric and positive definite;
  // 3. r1, r2, r3 are non-negative regularization parameters.
  //
  // NOTE: the user is responsible for storing H, C, G on their side.

  using FactorCallback = std::function<void(const double *w, const double r1,
                                            const double r2, const double r3)>;

  using SolveCallback = std::function<void(const double *b, double *v)>;

  using Block3x3KKTProductCallback = std::function<void(
      const double *w, const double r1, const double r2, const double r3,
      const double *x_x, const double *x_y, const double *x_z, double *y_x,
      double *y_y, double *y_z)>;

  using MatrixVectorMultiplicationCallback =
      std::function<void(const double *x, double *y)>;

  using ScalarGetter = std::function<double(void)>;

  using VectorGetter = std::function<const double *(void)>;

  using ModelCallback = std::function<void(const ModelCallbackInput &)>;

  using TimeoutCallback = std::function<bool(void)>;

  struct Dimensions {
    int x_dim;
    int s_dim;
    int y_dim;
  };

  // Callback for factoring the reduced-block-3x3 Newton-KKT system.
  FactorCallback factor;
  // Callback for solving the reduced-block-3x3 Newton-KKT system.
  SolveCallback solve;
  // Callback for y += Kx, where K is the block-3x3 Newton-KKT system's LHS.
  Block3x3KKTProductCallback add_Kx_to_y;
  // Callback for adding H x to y, where H is the Hessian of the Lagrangian.
  MatrixVectorMultiplicationCallback add_Hx_to_y;
  // Callback for adding C x to y, where C = J(c).
  MatrixVectorMultiplicationCallback add_Cx_to_y;
  // Callback for adding C^T x to y, where C = J(c).
  MatrixVectorMultiplicationCallback add_CTx_to_y;
  // Callback for adding G x to y, where G = J(g).
  MatrixVectorMultiplicationCallback add_Gx_to_y;
  // Callback for adding G^T x to y, where G = J(g).
  MatrixVectorMultiplicationCallback add_GTx_to_y;
  // Callback for getting the objective f(x).
  ScalarGetter get_f;
  // Callback for getting the gradient of the objective f(x).
  VectorGetter get_grad_f;
  // Callback for getting the equality constraint vector c(x).
  VectorGetter get_c;
  // Callback for getting the inequality constraint vector g(x).
  VectorGetter get_g;
  // Callback for evaluating the user model.
  ModelCallback model_callback;
  // Callback for (optionally) declaring a timeout. Return true for timeout.
  TimeoutCallback timeout_callback;
  // The problem dimensions.
  Dimensions dimensions;
};

struct Output {
  // The exit status of the optimization process.
  Status exit_status;
  int num_iterations;
  double max_primal_violation;
  double max_dual_violation;
};

struct VariablesWorkspace {
  // The primal variables.
  double *x;
  // The slack variables.
  double *s;
  // The dual variables associated with equality constraints.
  double *y;
  // The dual variables associated with inequality constraints.
  double *z;
  // The elastic variables.
  double *e;

  // To dynamically allocate the required memory.
  void reserve(int x_dim, int s_dim, int y_dim);
  void free();

  // For using pre-allocated (possibly statically allocated) memory.
  auto mem_assign(int x_dim, int s_dim, int y_dim, unsigned char *mem_ptr)
      -> int;

  // For knowing how much memory to pre-allocate.
  static constexpr auto num_bytes(int x_dim, int s_dim, int y_dim) -> int {
    return (x_dim + y_dim + 3 * s_dim) * sizeof(double);
  }
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

  // For knowing how much memory to pre-allocate.
  static constexpr auto num_bytes(int s_dim) -> int {
    return 2 * s_dim * sizeof(double);
  }
};

struct ComputeSearchDirectionWorkspace {
  // Stores S^{-1} Z
  double *w;
  // The RHS of the reduced block-3x3 Newton-KKT system.
  double *rhs_block_3x3;
  // The solution of the reduced block-3x3 Newton-KKT system.
  double *sol_block_3x3;
  // The solution of the iterative refinement system.
  double *iterative_refinement_error_sol;
  // Stores the residual of the full Newton-KKT system.
  double *residual;

  // To dynamically allocate the required memory.
  void reserve(int s_dim, int kkt_dim, int full_dim);
  void free();

  // For using pre-allocated (possibly statically allocated) memory.
  auto mem_assign(int s_dim, int kkt_dim, int full_dim, unsigned char *mem_ptr)
      -> int;

  // For knowing how much memory to pre-allocate.
  static constexpr auto num_bytes(int s_dim, int kkt_dim, int full_dim) -> int {
    return (s_dim + 3 * kkt_dim + full_dim) * sizeof(double);
  }
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
  // The delta variable storage (for both primal and dual variables).
  VariablesWorkspace delta_vars;
  // The next variable storage (for both primal and dual variables).
  VariablesWorkspace next_vars;
  // The negative Newton-KKT RHS storage (for both primal and dual variables).
  VariablesWorkspace nrhs;
  // Stores miscellaneous items.
  MiscellaneousWorkspace miscellaneous_workspace;
  // Stores the workspace used in compute_search_direction.
  ComputeSearchDirectionWorkspace csd_workspace;

  // To dynamically allocate the required memory.
  void reserve(int x_dim, int s_dim, int y_dim);
  void free();

  // For using pre-allocated (possibly statically allocated) memory.
  auto mem_assign(int x_dim, int s_dim, int y_dim, unsigned char *mem_ptr)
      -> int;

  // For knowing how much memory to pre-allocate.
  static constexpr auto num_bytes(int x_dim, int s_dim, int y_dim) -> int {
    const int kkt_dim = x_dim + s_dim + y_dim;
    const int full_dim = kkt_dim + s_dim + s_dim;
    return 4 * VariablesWorkspace::num_bytes(x_dim, s_dim, y_dim) +
           MiscellaneousWorkspace::num_bytes(s_dim) +
           ComputeSearchDirectionWorkspace::num_bytes(s_dim, kkt_dim, full_dim);
  }
};

} // namespace sip
