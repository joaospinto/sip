#include "sip/helpers.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

namespace sip {

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

auto weighted_dot(const double *x, const double *w, const double *y,
                  const int dim) -> double {
  double out = 0.0;
  for (int i = 0; i < dim; ++i) {
    out += w[i] * x[i] * y[i];
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

auto weighted_squared_norm(const double *x, const double *w, const int dim)
    -> double {
  double out = 0.0;
  for (int i = 0; i < dim; ++i) {
    out += w[i] * x[i] * x[i];
  }
  return out;
}

auto norm(const double *x, const int dim) -> double {
  return std::sqrt(squared_norm(x, dim));
}

auto inf_norm(const double *x, const int dim) -> double {
  double out = 0.0;
  for (int i = 0; i < dim; ++i) {
    out = std::max(out, std::fabs(x[i]));
  }
  return out;
}

auto max_abs_or_inf(const double *x, const int dim) -> double {
  double out = 0.0;
  for (int i = 0; i < dim; ++i) {
    const double abs_x = std::fabs(x[i]);
    out = std::isnan(abs_x) ? std::numeric_limits<double>::infinity()
                            : std::max(out, abs_x);
  }
  return out;
}

auto max_positive_or_inf(const double *x, const int dim) -> double {
  double out = 0.0;
  for (int i = 0; i < dim; ++i) {
    out = std::isnan(x[i]) ? std::numeric_limits<double>::infinity()
                           : std::max(out, x[i]);
  }
  return out;
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
