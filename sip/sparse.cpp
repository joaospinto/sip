#include "sparse.hpp"

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
  ::free(ind);
  ::free(indptr);
  ::free(data);
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

auto add(const SparseMatrix &A, const SparseMatrix &B,
         SparseMatrix &C) -> void {
  assert(A.rows == B.rows);
  assert(A.cols == B.cols);
  assert(A.is_transposed == B.is_transposed);
  C.is_transposed = A.is_transposed;
  C.rows = A.rows;
  C.cols = A.cols;
  int idx{0};
  C.indptr[0] = idx;
  for (int j = 0; j < C.cols; ++j) {
    int it_A = A.indptr[j];
    int it_B = B.indptr[j];

    while (it_A < A.indptr[j + 1] || it_B < B.indptr[j + 1]) {
      if (it_A < A.indptr[j + 1] &&
          (A.ind[it_A] < B.ind[it_B] || it_B == B.indptr[j + 1])) {
        C.ind[idx] = A.ind[it_A];
        C.data[idx] = A.data[it_A];
        ++it_A;
        ++idx;
      } else if (it_B < B.indptr[j + 1] &&
                 (A.ind[it_A] > B.ind[it_B] || it_A == A.indptr[j + 1])) {
        C.ind[idx] = B.ind[it_B];
        C.data[idx] = B.data[it_B];
        ++it_B;
        ++idx;
      } else {
        C.ind[idx] = A.ind[it_A];
        C.data[idx] = A.data[it_A] + B.data[it_B];
        ++it_A;
        ++it_B;
        ++idx;
      }
    }
    C.indptr[j + 1] = idx;
  }
}

auto sparse_weighted_dot(int x_ind_size, const int *x_ind, const double *x_data,
                         const double *weights, int y_ind_size,
                         const int *y_ind,
                         const double *y_data) -> std::pair<double, bool> {
  int it_x{0};
  int it_y{0};
  double out{0.0};
  bool potentially_nz = false;
  while (it_x < x_ind_size && it_y < y_ind_size) {
    if (x_ind[it_x] < y_ind[it_y]) {
      ++it_x;
    } else if (x_ind[it_x] > y_ind[it_y]) {
      ++it_y;
    } else {
      potentially_nz = true;
      out += x_data[it_x] * y_data[it_y] * weights[x_ind[it_x]];
      ++it_x;
      ++it_y;
    }
  }
  return {out, potentially_nz};
}

auto XT_D_X(const SparseMatrix &X, double *D, SparseMatrix &XT_D_X) -> void {
  XT_D_X.rows = X.cols;
  XT_D_X.cols = X.cols;
  XT_D_X.is_transposed = false;
  int idx{0};
  XT_D_X.indptr[0] = idx;
  for (int j = 0; j < X.cols; ++j) {
    for (int i = 0; i <= j; ++i) {
      const auto [dot, dot_potentially_nz] = sparse_weighted_dot(
          X.indptr[i + 1] - X.indptr[i], X.ind + X.indptr[i],
          X.data + X.indptr[i], D, X.indptr[j + 1] - X.indptr[j],
          X.ind + X.indptr[j], X.data + X.indptr[j]);
      if (dot_potentially_nz) {
        XT_D_X.ind[idx] = i;
        XT_D_X.data[idx] = dot;
        ++idx;
      }
    }
    XT_D_X.indptr[j + 1] = idx;
  }
}

auto _add_ATx_to_y_impl(const SparseMatrix &A, const double *x, double *y) {
  for (int j = 0; j < A.cols; ++j) {
    for (int i = A.indptr[j]; i < A.indptr[j + 1]; ++i) {
      y[j] += A.data[i] * x[A.ind[i]];
    }
  }
}

auto _add_Ax_to_y_impl(const SparseMatrix &A, const double *x,
                       double *y) -> void {
  for (int j = 0; j < A.cols; j++) {
    const int value_idx_end = A.indptr[j + 1];
    for (int value_idx = A.indptr[j]; value_idx < value_idx_end; value_idx++) {
      const int i = A.ind[value_idx];
      y[i] += A.data[value_idx] * x[j];
    }
  }
}

auto _add_weighted_ATx_to_y_impl(const SparseMatrix &A, const double *weights,
                                 const double *x, double *y) {
  for (int j = 0; j < A.cols; ++j) {
    for (int i = A.indptr[j]; i < A.indptr[j + 1]; ++i) {
      y[j] += weights[j] * A.data[i] * x[A.ind[i]];
    }
  }
}

auto _add_weighted_Ax_to_y_impl(const SparseMatrix &A, const double *weights,
                                const double *x, double *y) -> void {
  for (int j = 0; j < A.cols; j++) {
    const int value_idx_end = A.indptr[j + 1];
    for (int value_idx = A.indptr[j]; value_idx < value_idx_end; value_idx++) {
      const int i = A.ind[value_idx];
      y[i] += weights[i] * A.data[value_idx] * x[j];
    }
  }
}

auto add(const double *x, const double *y, const int dim, double *z) -> void {
  for (int i = 0; i < dim; ++i) {
    z[i] = x[i] + y[i];
  }
}

auto add_ATx_to_y(const SparseMatrix &A, const double *x, double *y) -> void {
  if (A.is_transposed) {
    _add_Ax_to_y_impl(A, x, y);
  } else {
    _add_ATx_to_y_impl(A, x, y);
  }
}

auto add_Ax_to_y(const SparseMatrix &A, const double *x, double *y) -> void {
  if (A.is_transposed) {
    _add_ATx_to_y_impl(A, x, y);
  } else {
    _add_Ax_to_y_impl(A, x, y);
  }
}

auto add_weighted_Ax_to_y(const SparseMatrix &A, const double *weights,
                          const double *x, double *y) -> void {
  if (A.is_transposed) {
    _add_weighted_ATx_to_y_impl(A, weights, x, y);
  } else {
    _add_weighted_Ax_to_y_impl(A, weights, x, y);
  }
}

auto add_Ax_to_y_where_A_upper_symmetric(const SparseMatrix &A, const double *x,
                                         double *y) -> void {
  _add_ATx_to_y_impl(A, x, y);
  _add_Ax_to_y_impl(A, x, y);
  for (int j = 0; j < A.cols; ++j) {
    const int value_idx_end = A.indptr[j + 1];
    for (int value_idx = A.indptr[j]; value_idx < value_idx_end; value_idx++) {
      const int i = A.ind[value_idx];
      if (i == j) {
        y[i] -= A.data[value_idx] * x[j];
      }
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
