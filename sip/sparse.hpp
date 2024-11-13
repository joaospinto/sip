#include <ostream>

namespace sip {

struct SparseMatrix {
  // The number of rows of the matrix.
  int rows;
  // The number of cols of the matrix.
  int cols;
  // The row indices of each entry.
  int *ind;
  // The column start indices of each column.
  int *indptr;
  // The potentially non-zero entries.
  double *data;
  // Whether the matrix is transposed.
  bool is_transposed;

  // NOTE: the user may also direct pointers to statically allocated memory.
  auto reserve(int dim, int nnz) -> void;
  auto free() -> void;
};

// Useful for debugging.
auto operator<<(std::ostream &os, const SparseMatrix &M) -> std::ostream &;

auto add(const SparseMatrix &A, const SparseMatrix &B, SparseMatrix &C) -> void;

auto XT_D_X(const SparseMatrix &X, double *D, SparseMatrix &XT_D_X) -> void;

auto add(const double *x, const double *y, const int dim, double *z) -> void;

auto add_ATx_to_y(const SparseMatrix &A, const double *x, double *y) -> void;

auto add_Ax_to_y(const SparseMatrix &A, const double *x, double *y) -> void;

auto add_weighted_Ax_to_y(const SparseMatrix &A, const double *weights,
                          const double *x, double *y) -> void;

auto add_Ax_to_y_where_A_upper_symmetric(const SparseMatrix &A, const double *x,
                                         double *y) -> void;

auto dot(const double *x, const double *y, const int dim) -> double;

auto sum_of_logs(const double *x, const int dim) -> double;

auto min_element_product(const double *x, const double *y,
                         const int dim) -> double;

auto squared_norm(const double *x, const int dim) -> double;

auto norm(const double *x, const int dim) -> double;

auto x_dot_y_inverse(const double *x, const double *y, const int dim) -> double;

} // namespace sip
