#pragma once

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

  // To dynamically allocate the required memory.
  auto reserve(int dim, int nnz) -> void;
  auto free() -> void;

  // For using pre-allocated (possibly statically allocated) memory.
  auto mem_assign(int dim, int nnz, unsigned char *mem_ptr) -> int;
};

// Useful for debugging.
auto operator<<(std::ostream &os, const SparseMatrix &M) -> std::ostream &;

auto add(const double *x, const double *y, const int dim, double *z) -> void;

auto dot(const double *x, const double *y, const int dim) -> double;

auto sum_of_logs(const double *x, const int dim) -> double;

auto min_element_product(const double *x, const double *y, const int dim)
    -> double;

auto squared_norm(const double *x, const int dim) -> double;

auto norm(const double *x, const int dim) -> double;

auto x_dot_y_inverse(const double *x, const double *y, const int dim) -> double;

} // namespace sip
