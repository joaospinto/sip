#pragma once

namespace sip {

auto add(const double *x, const double *y, const int dim, double *z) -> void;

auto dot(const double *x, const double *y, const int dim) -> double;

auto sum_of_logs(const double *x, const int dim) -> double;

auto min_element_product(const double *x, const double *y, const int dim)
    -> double;

auto squared_norm(const double *x, const int dim) -> double;

auto norm(const double *x, const int dim) -> double;

auto inf_norm(const double *x, const int dim) -> double;

auto x_dot_y_inverse(const double *x, const double *y, const int dim) -> double;

} // namespace sip
