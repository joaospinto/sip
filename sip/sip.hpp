#include <fstream>
#include <functional>

#include "sparse.hpp"

namespace sip {

enum class Status {
  SOLVED = 0,
  ITERATION_LIMIT = 1,
  LINE_SEARCH_FAILURE = 2,
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
  SparseMatrix jacobian_c_transpose;

  // The inequality constraints and their first derivative.
  double *g;
  SparseMatrix jacobian_g_transpose;

  // NOTE: the user may also direct pointers to statically allocated memory.
  void reserve(int x_dim, int s_dim, int y_dim, int upper_hessian_f_nnz,
               int jacobian_c_nnz, int jacobian_g_nnz);
  void free();
};

struct Input {
  // Callback for filling the ModelCallbackOutput object.
  std::function<void(const ModelCallbackInput &, ModelCallbackOutput &)>
      model_callback;
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
  // Determines whether we should print the solver logs.
  bool print_logs = true;
};

struct Output {
  // The exit status of the optimization process.
  Status exit_status;
};

struct QDLDLWorkspace {
  // Definitions:
  // dim represents the dimension of the KKT system,
  //     i.e. dim = x_dim + y_dim + 2 * s_dim.
  // L_nnz represents the number of non-zeros in the L matrix,
  //     to be determined by the user.

  // Elimination tree workspace.
  int *etree; // Required size: dim
  int *Lnz;   // Required size: dim

  // Factorization workspace.
  int *iwork;           // Required size: 3 * dim
  unsigned char *bwork; // Required size: dim
  double *fwork;        // Required size: dim

  // Factorizaton output storage.
  int *Lp;      // Required size: L_nnz
  int *Li;      // Required size: L_nnz
  double *Lx;   // Required size: L_nnz
  double *D;    // Required size: dim
  double *Dinv; // Required size: dim

  // Solve workspace.
  double *x; // Required size: dim

  // NOTE: the user may also direct pointers to statically allocated memory.
  void reserve(int kkt_dim, int kkt_L_nnz);
  void free();
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
  // The next primal variables.
  double *next_x;
  // The next slack variables.
  double *next_s;

  // NOTE: the user may also direct pointers to statically allocated memory.
  void reserve(int x_dim, int s_dim, int y_dim);
  void free();
};

struct MiscellaneousWorkspace {
  // Stores g(x) + s.
  double *g_plus_s;

  // NOTE: the user may also direct pointers to statically allocated memory.
  void reserve(int s_dim);
  void free();
};

struct KKTWorkspace {
  // The LHS of the KKT system (requires size hess_f_nnz + jac_c_nnz + jac_g_nnz
  // + 3 * s_dim + y_dim).
  SparseMatrix lhs;
  // The (negative) RHS of the KKT system (requires size x_dim + y_dim + 2 *
  // s_dim).
  double *negative_rhs;

  // NOTE: the user may also direct pointers to statically allocated memory.
  void reserve(int kkt_dim, int kkt_nnz);
  void free();
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

  // NOTE: the user may also direct pointers to statically allocated memory.
  void reserve(int x_dim, int s_dim, int y_dim, int upper_hessian_f_nnz,
               int jacobian_c_nnz, int jacobian_g_nnz, int kkt_L_nnz);
  void free();
};

auto solve(const Input &input, const Settings &settings, Workspace &workspace,
           Output &output) -> void;

} // namespace sip
