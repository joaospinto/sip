#include <functional>
#include <ostream>

#include "sparse.hpp"

namespace sip {

enum class Status {
  SOLVED = 0,
  ITERATION_LIMIT = 1,
  LINE_SEARCH_FAILURE = 2,
};

struct Settings {
  // Determines how the Newton-KKT system is solved.
  enum class LinearSystemFormulation {
    SYMMETRIC_DIRECT_4x4 = 0,
    // TODO(joao): add support for the 3x3 version.
    // SYMMETRIC_INDIRECT_3x3 = 1,
    SYMMETRIC_INDIRECT_2x2 = 2,
  };
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
  // Determines how the search direction is computed.
  LinearSystemFormulation lin_sys_formulation =
      LinearSystemFormulation::SYMMETRIC_INDIRECT_2x2;
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
  // NOTE: hessian_f should be a positive semi-definite approximation.
  // NOTE: only the upper triangle should be filled in hessian_f.
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
  void reserve(Settings::LinearSystemFormulation lin_sys_formulation, int x_dim, int s_dim, int y_dim, int upper_hessian_f_nnz,
               int jacobian_c_nnz, int jacobian_g_nnz);
  void free();

  // For using pre-allocated (possibly statically allocated) memory.
  auto mem_assign(Settings::LinearSystemFormulation lin_sys_formulation, int x_dim, int s_dim, int y_dim, int upper_hessian_f_nnz,
                  int jacobian_c_nnz, int jacobian_g_nnz,
                  unsigned char* mem_ptr) -> int;
};

struct Input {
  // Callback for filling the ModelCallbackOutput object.
  std::function<void(const ModelCallbackInput &, ModelCallbackOutput &)>
      model_callback;
};

struct Output {
  // The exit status of the optimization process.
  Status exit_status;
};

struct QDLDLWorkspace {
  // Elimination tree workspace.
  int *etree; // Required size: kkt_dim
  int *Lnz;   // Required size: kkt_dim

  // Factorization workspace.
  int *iwork;           // Required size: 3 * kkt_dim
  unsigned char *bwork; // Required size: kkt_dim
  double *fwork;        // Required size: kkt_dim

  // Factorizaton output storage.
  int *Lp;      // Required size: kkt_dim + 1
  int *Li;      // Required size: kkt_L_nnz
  double *Lx;   // Required size: kkt_L_nnz
  double *D;    // Required size: kkt_dim
  double *Dinv; // Required size: kkt_dim

  // Solve workspace.
  double *x; // Required size: kkt_dim

  // To dynamically allocate the required memory.
  void reserve(int kkt_dim, int kkt_L_nnz);
  void free();

  // For using pre-allocated (possibly statically allocated) memory.
  auto mem_assign(int kkt_dim, int kkt_L_nnz,
                  unsigned char* mem_ptr) -> int;
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
  // The candidate delta in the slack variables.
  double *ds;
  // The candidate delta in the dual variables associated with inequalities.
  double *dz;
  // The candidate delta in the elastic variables.
  double *de;

  // To dynamically allocate the required memory.
  void reserve(int x_dim, int s_dim, int y_dim);
  void free();

  // For using pre-allocated (possibly statically allocated) memory.
  auto mem_assign(int x_dim, int s_dim, int y_dim,
                  unsigned char* mem_ptr) -> int;
};

struct MiscellaneousWorkspace {
  // Stores g(x) + s.
  double *g_plus_s;
  // Stores g(x) + s (+ e, when applicable).
  double *g_plus_s_plus_e;
  // Stores the linear system residual.
  double *lin_sys_residual;
  // Stores the x-gradient of the Lagrangian.
  double *grad_x_lagrangian;
  // Stores sigma = z / (s + gamma_z * z).
  double *sigma;
  // Stores sigma * (g(x) + (mu / z)).
  double *sigma_times_g_plus_mu_over_z_minus_z_over_p;
  // Stores jacobian_g_t @ sigma @ jacobian_g.
  SparseMatrix jac_g_t_sigma_jac_g;

  // To dynamically allocate the required memory.
  void reserve(int x_dim, int s_dim, int kkt_dim, int jac_g_t_jac_g_nnz);
  void free();

  // For using pre-allocated (possibly statically allocated) memory.
  auto mem_assign(int x_dim, int s_dim, int kkt_dim, int jac_g_t_jac_g_nnz,
                  unsigned char* mem_ptr) -> int;
};

struct KKTWorkspace {
  // The LHS of the (potentially reduced/eliminated) KKT system.
  SparseMatrix lhs;
  // The (negative) RHS of the (potentially reduced/eliminated )KKT system.
  double *negative_rhs;

  // To dynamically allocate the required memory.
  void reserve(int kkt_dim, int kkt_nnz);
  void free();

  // For using pre-allocated (possibly statically allocated) memory.
  auto mem_assign(int kkt_dim, int kkt_nnz,
                  unsigned char* mem_ptr) -> int;
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
  // Storage of the LHS and RHS of the KKT system.
  KKTWorkspace kkt_workspace;
  // The workspace of the QDLDL solver.
  QDLDLWorkspace qdldl_workspace;
  // The model callback workspace.
  ModelCallbackOutput model_callback_output;
  // Stores miscellaneous items.
  MiscellaneousWorkspace miscellaneous_workspace;

  // To dynamically allocate the required memory.
  void reserve(Settings::LinearSystemFormulation lin_sys_formulation, int x_dim,
               int s_dim, int y_dim, int upper_hessian_f_nnz,
               int jacobian_c_nnz, int jac_g_t_jac_g_nnz, int jacobian_g_nnz,
               int upper_hessian_f_plus_upper_jac_g_t_jac_g_nnz, int kkt_L_nnz);
  void free();

  // For using pre-allocated (possibly statically allocated) memory.
  auto mem_assign(Settings::LinearSystemFormulation lin_sys_formulation, int x_dim,
                  int s_dim, int y_dim, int upper_hessian_f_nnz,
                  int jacobian_c_nnz, int jac_g_t_jac_g_nnz, int jacobian_g_nnz,
                  int upper_hessian_f_plus_upper_jac_g_t_jac_g_nnz, int kkt_L_nnz,
                  unsigned char* mem_ptr) -> int;
};

auto solve(const Input &input, const Settings &settings, Workspace &workspace,
           Output &output) -> void;

} // namespace sip
