#include "helpers.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

namespace sip {

auto SparseMatrix::reserve(int dim, int nnz) -> void {
  ind = new int[nnz];
  indptr = new int[dim + 1];
  data = new double[nnz];
}

auto SparseMatrix::free() -> void {
  delete[] ind;
  delete[] indptr;
  delete[] data;
}

auto SparseMatrix::mem_assign(int dim, int nnz, unsigned char *mem_ptr) -> int {
  int cum_size = 0;
  ind = reinterpret_cast<decltype(ind)>(mem_ptr + cum_size);
  cum_size += nnz * sizeof(int);
  indptr = reinterpret_cast<decltype(indptr)>(mem_ptr + cum_size);
  cum_size += (dim + 1) * sizeof(int);
  data = reinterpret_cast<decltype(data)>(mem_ptr + cum_size);
  cum_size += nnz * sizeof(double);
  return cum_size;
}

auto operator<<(std::ostream &os, const SparseMatrix &M) -> std::ostream & {
  os << "rows: " << M.rows;
  os << "\ncols: " << M.cols;
  os << "\nindptr: ";
  for (int i = 0; i <= M.cols; ++i) {
    os << M.indptr[i];
    if (i < M.cols) {
      os << ", ";
    }
  }
  const int nnz = M.indptr[M.cols];
  os << "\nind: ";
  for (int i = 0; i < nnz; ++i) {
    os << M.ind[i];
    if (i + 1 < nnz) {
      os << ", ";
    }
  }
  os << "\ndata: ";
  for (int i = 0; i < nnz; ++i) {
    os << M.data[i];
    if (i + 1 < nnz) {
      os << ", ";
    }
  }
  os << "\nis_transposed: " << M.is_transposed;
  return os;
}

auto add(const double *x, const double *y, const int dim, double *z) -> void {
  for (int i = 0; i < dim; ++i) {
    z[i] = x[i] + y[i];
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

auto min_element_product(const double *x, const double *y, const int dim)
    -> double {
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

auto x_dot_y_inverse(const double *x, const double *y, const int dim)
    -> double {
  double out = 0.0;
  for (int i = 0; i < dim; ++i) {
    out += x[i] / y[i];
  }
  return out;
}

} // namespace sip
