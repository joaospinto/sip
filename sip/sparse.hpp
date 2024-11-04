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
};

struct DenseVector {
  // The dimension of the vector.
  int dim;
  // The vector data storage.
  double *data;
};

auto add_ATx_to_y(const SparseMatrix &A, double *x, double *y) -> void;

} // namespace sip
