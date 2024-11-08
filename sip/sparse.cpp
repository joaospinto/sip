#include "sparse.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace sip {

void SparseMatrix::reserve(int dim, int nnz) {
  ind = new int[nnz];
  indptr = new int[dim + 1];
  data = new double[nnz];
}

void SparseMatrix::free() {
  ::free(ind);
  ::free(indptr);
  ::free(data);
}

auto add(const double *x, const double *y, const int dim, double *z) -> void {
  for (int i = 0; i < dim; ++i) {
    z[i] = x[i] + y[i];
  }
}

auto add_ATx_to_y(const SparseMatrix &A, const double *x, double *y) -> void {
  for (int j = 0; j < A.cols; ++j) {
    for (int i = A.indptr[j]; i < A.indptr[j + 1]; ++i) {
      y[j] += A.data[i] * x[A.ind[i]];
    }
  }
}

auto add_Ax_to_y(const SparseMatrix &A, const double *x, double *y) -> void {
  for (int j = 0; j < A.cols; j++) {
    const int value_idx_end = A.indptr[j + 1];
    for (int value_idx = A.indptr[j]; value_idx < value_idx_end; value_idx++) {
      const int i = A.ind[value_idx];
      y[i] += A.data[value_idx] * x[j];
    }
  }
}

auto dot(const double *x, const double *y, const int dim) -> double {
  double out = 0.0;
  for (int i = 0; i < dim; ++i) {
    out += x[i] * y[i];
  }
  return out;
}

auto sum_of_logs(const double *x, const int dim) -> double {
  double out = 0.0;
  for (int i = 0; i < dim; ++i) {
    out += std::log(x[i]);
  }
  return out;
}

auto min_element_product(const double *x, const double *y,
                         const int dim) -> double {
  double out = std::numeric_limits<double>::infinity();
  for (int i = 0; i < dim; ++i) {
    out = std::min(out, x[i] * y[i]);
  }
  return out;
}

auto squared_norm(const double *x, const int dim) -> double {
  double out = 0.0;
  for (int i = 0; i < dim; ++i) {
    out += x[i] * x[i];
  }
  return out;
}

auto norm(const double *x, const int dim) -> double {
  return std::sqrt(squared_norm(x, dim));
}

auto x_dot_y_inverse(const double *x, const double *y,
                     const int dim) -> double {
  double out = 0.0;
  for (int i = 0; i < dim; ++i) {
    out += x[i] / y[i];
  }
  return out;
}

} // namespace sip
