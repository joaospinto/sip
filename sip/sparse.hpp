namespace sip {

struct SparseMatrix {
  // The number of rows of the matrix.
  int rows;
  // The number of cols of the matrix.
  int cols;
  // The number of potentially non-zero entries of the matrix.
  int nnz;
  // The row indices of each entry.
  int *ind;
  // The column start indices of each column.
  int *indptr;
  // The potentially non-zero entries.
  double *data;

  // NOTE: the user may also direct pointers to statically allocated memory.
  void reserve(int dim, int nnz);
  void free();
};

auto add(const double *x, const double *y, const int dim, double *z) -> void;

auto add_ATx_to_y(const SparseMatrix &A, const double *x, double *y) -> void;

auto add_Ax_to_y(const SparseMatrix &A, const double *x, double *y) -> void;

auto dot(const double *x, const double *y, const int dim) -> double;

auto sum_of_logs(const double *x, const int dim) -> double;

auto min_element_product(const double *x, const double *y,
                         const int dim) -> double;

auto squared_norm(const double *x, const int dim) -> double;

auto norm(const double *x, const int dim) -> double;

auto x_dot_y_inverse(const double *x, const double *y, const int dim) -> double;

} // namespace sip
