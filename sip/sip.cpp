#include "sip.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <fmt/core.h>
#include <iostream>
#include <limits>
#include <tuple>
#include <utility>

namespace sip {

auto get_s_dim(const SparseMatrix &jacobian_g) -> int {
  return jacobian_g.is_transposed ? jacobian_g.cols : jacobian_g.rows;
}

auto get_y_dim(const SparseMatrix &jacobian_c) -> int {
  return jacobian_c.is_transposed ? jacobian_c.cols : jacobian_c.rows;
}

auto get_max_step_sizes(const int s_dim, const double tau, const double *s,
                        const double *z, const double *ds, const double *dz)
    -> std::pair<double, double> {
  // s + alpha_s_max * ds >= (1 - tau) * s
  // z + alpha_z_max * dz >= (1 - tau) * z

  double alpha_s_max = 1.0;
  double alpha_z_max = 1.0;

  for (int i = 0; i < s_dim; ++i) {
    if (ds[i] < 0.0) {
      alpha_s_max = std::min(alpha_s_max, tau * s[i] / std::max(-ds[i], 1e-16));
    }
    if (dz[i] < 0.0) {
      alpha_z_max = std::min(alpha_z_max, tau * z[i] / std::max(-dz[i], 1e-16));
    }
  }

  return {alpha_s_max, alpha_z_max};
}

auto adaptive_mu(const int s_dim, double *s, double *z) -> double {
  // Uses the LOQO rule mentioned in Nocedal & Wright.
  const double dp = dot(s, z, s_dim);
  const double zeta = min_element_product(s, z, s_dim) * s_dim / dp;
  const auto cube = [](const double k) { return k * k * k; };
  const double sigma = 0.1 * cube(std::min(0.5 * (1.0 - zeta) / zeta, 2.0));
  return sigma * dp / s_dim;
}

auto barrier_lagrangian_slope(const Settings &settings,
                              const Workspace &workspace, const double *dx,
                              const double *ds, const double *dy,
                              const double *dz, const double *de)
    -> std::pair<double, double> {
  const auto &mco = *workspace.model_callback_output;
  const int x_dim = mco.upper_hessian_f.rows;
  const int s_dim = get_s_dim(mco.jacobian_g);
  const int y_dim = get_y_dim(mco.jacobian_c);
  const double x_slope = dot(workspace.nrhs.x, dx, x_dim);
  const double s_slope = dot(workspace.nrhs.s, ds, s_dim);
  const double y_slope = dot(workspace.nrhs.y, dy, y_dim);
  const double z_slope = dot(workspace.nrhs.z, dz, s_dim);
  const double e_slope =
      settings.enable_elastics
          ? settings.elastic_var_cost_coeff * dot(workspace.nrhs.e, de, s_dim)
          : 0.0;
  const double bl_slope = x_slope + s_slope + y_slope + z_slope + e_slope;
  const double max_neg_slope =
      std::max(-std::min({x_slope, s_slope, y_slope, z_slope, e_slope}), 0.0);
  return std::make_pair(bl_slope, max_neg_slope);
}

auto get_rho(const double max_barrier_lagrangian_slope_for_zero_rho,
             const double max_rho, const double barrier_lagrangian_slope,
             const double penalty_multiplier_slope, const double k) -> double {
  // NOTE: D(merit_function; dx, ds, dy, dz, de) = barrier_lagrangian_slope
  //           - rho * penalty_multiplier_slope.
  //
  // Moreover, for any k > 0:
  //                               (barrier_lagrangian_slope + k)
  // merit_slope <= -k iff rho >= -------------------------------- .
  //                                  penalty_multiplier_slope
  if (barrier_lagrangian_slope < max_barrier_lagrangian_slope_for_zero_rho) {
    return 0.0;
  }
  return std::clamp((barrier_lagrangian_slope + k) / penalty_multiplier_slope,
                    0.0, max_rho);
}

auto merit_function(const Settings &settings, const Workspace &workspace,
                    const double *s, const double *y, const double *z,
                    const double *e, const double mu, const double rho,
                    const double sq_constraint_violation_norm)
    -> std::tuple<double, double, double, double, double, double, double> {
  const auto &mco = *workspace.model_callback_output;
  const int s_dim = get_s_dim(mco.jacobian_g);
  const int y_dim = get_y_dim(mco.jacobian_c);
  const double s_term = -mu * sum_of_logs(s, s_dim);
  const double c_term = dot(mco.c, y, y_dim);
  const double g_term =
      dot(workspace.miscellaneous_workspace.g_plus_s_plus_e, z, s_dim);
  const double e_term =
      settings.enable_elastics
          ? 0.5 * settings.elastic_var_cost_coeff * squared_norm(e, s_dim)
          : 0.0;
  const double barrier_lagrangian = mco.f + s_term + c_term + g_term + e_term;
  const double aug_term = 0.5 * rho * sq_constraint_violation_norm;
  const double merit = barrier_lagrangian + aug_term;
  return std::make_tuple(mco.f, s_term, c_term, g_term, e_term, aug_term,
                         merit);
}

auto compute_search_direction(const Input &input, const Settings &settings,
                              const double gamma_y, const double gamma_z,
                              const double mu, Workspace &workspace)
    -> std::tuple<const double *, const double *, const double *,
                  const double *, const double *, double, double> {
  const double p = settings.enable_elastics
                       ? settings.elastic_var_cost_coeff
                       : std::numeric_limits<double>::infinity();
  const auto &mco = *workspace.model_callback_output;
  constexpr double gamma_x = 0.0;

  double kkt_error;
  double lin_sys_error;
  double *dx = workspace.delta_vars.x;
  double *ds = workspace.delta_vars.s;
  double *dy = workspace.delta_vars.y;
  double *dz = workspace.delta_vars.z;
  double *de = workspace.delta_vars.e;
  double *rx = workspace.nrhs.x;
  double *rs = workspace.nrhs.s;
  double *ry = workspace.nrhs.y;
  double *rz = workspace.nrhs.z;
  double *re = workspace.nrhs.e;

  input.lin_sys_solver(mco.c, mco.g, mco.gradient_f, mco.upper_hessian_f.data,
                       mco.jacobian_c.data, mco.jacobian_g.data,
                       workspace.vars.s, workspace.vars.y, workspace.vars.z,
                       workspace.vars.e, mu, p, gamma_x, gamma_y, gamma_z, dx,
                       ds, dy, dz, de, rx, rs, ry, rz, re, kkt_error,
                       settings.print_logs ? &lin_sys_error : nullptr);

  return std::make_tuple(dx, ds, dy, dz, de, kkt_error, lin_sys_error);
}

auto check_settings([[maybe_unused]] const Settings &settings) {
  assert(!settings.enable_elastics || settings.elastic_var_cost_coeff > 0.0);
}

auto solve(const Input &input, const Settings &settings, Workspace &workspace,
           Output &output) -> void {
  {
    ModelCallbackInput mci{
        .x = workspace.vars.x,
    };

    input.model_callback(mci, &workspace.model_callback_output);
  }

  check_settings(settings);

  const int x_dim = workspace.model_callback_output->upper_hessian_f.cols;
  const int s_dim = get_s_dim(workspace.model_callback_output->jacobian_g);
  const int y_dim = get_y_dim(workspace.model_callback_output->jacobian_c);

  for (int i = 0; i < s_dim; ++i) {
    assert(workspace.vars.s[i] > 0.0 && workspace.vars.z[i] > 0.0);
  }

  add(workspace.model_callback_output->g, workspace.vars.s, s_dim,
      workspace.miscellaneous_workspace.g_plus_s);

  if (settings.enable_elastics) {
    add(workspace.miscellaneous_workspace.g_plus_s, workspace.vars.e, s_dim,
        workspace.miscellaneous_workspace.g_plus_s_plus_e);
  } else {
    std::copy(workspace.miscellaneous_workspace.g_plus_s,
              workspace.miscellaneous_workspace.g_plus_s + s_dim,
              workspace.miscellaneous_workspace.g_plus_s_plus_e);
  }

  if (settings.print_logs) {
    std::cout << fmt::format(
                     // clang-format off
                     "{:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10}",
                     // clang-format on
                     "iteration", "alpha", "merit", "f", "|c|", "|g+s+e|",
                     "bl_slope", "m_slope", "alpha_s_m", "alpha_z_m", "|dx|",
                     "|ds|", "|dy|", "|dz|", "|de|", "mu", "rho", "linsys_res",
                     "kkt_error")
              << std::endl;
  }

  for (int iteration = 0; iteration < settings.max_iterations; ++iteration) {
    const double mu =
        std::max(adaptive_mu(s_dim, workspace.vars.s, workspace.vars.z),
                 settings.mu_min);

    const double f0 = workspace.model_callback_output->f;

    const double ctc = squared_norm(workspace.model_callback_output->c, y_dim);
    const double gsetgse =
        squared_norm(workspace.miscellaneous_workspace.g_plus_s_plus_e, s_dim);

    const double sq_constraint_violation_norm = ctc + gsetgse;

    const double cty =
        dot(workspace.model_callback_output->c, workspace.vars.y, y_dim);
    const double gsetz = dot(workspace.miscellaneous_workspace.g_plus_s_plus_e,
                             workspace.vars.z, s_dim);

    const double gamma_y = cty > 0.0
                               ? std::min(0.5 * ctc / cty, settings.gamma_y)
                               : settings.gamma_y;
    const double gamma_z =
        gsetz > 0.0 ? std::min(0.5 * gsetgse / gsetz, settings.gamma_z)
                    : settings.gamma_z;

    const double penalty_multiplier_slope =
        sq_constraint_violation_norm - gamma_y * cty - gamma_z * gsetz;

    const auto [dx, ds, dy, dz, de, kkt_error, lin_sys_error] =
        compute_search_direction(input, settings, gamma_y, gamma_z, mu,
                                 workspace);

    const auto [bl_slope, max_neg_slope] =
        barrier_lagrangian_slope(settings, workspace, dx, ds, dy, dz, de);

    if (kkt_error < settings.max_kkt_violation) {
      if (settings.print_logs) {
        const double de_norm =
            settings.enable_elastics ? norm(de, s_dim) : -1.0;
        std::cout << fmt::format(
                         // clang-format off
                         "{:^+10} {:^10} {:^10} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^10} {:^10} {:^10} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^10} {:^+10.4g} {:^+10.4g}",
                         // clang-format on
                         iteration, "", "", workspace.model_callback_output->f,
                         std::sqrt(ctc), std::sqrt(gsetgse), bl_slope, "", "",
                         "", norm(dx, x_dim), norm(ds, s_dim), norm(dy, y_dim),
                         norm(dz, s_dim), de_norm, mu, "", lin_sys_error,
                         kkt_error)
                  << std::endl;
      }

      output.exit_status = Status::SOLVED;
      output.num_iterations = iteration;
      return;
    }

    const double tau = std::max(settings.tau_min, mu > 0.0 ? 1.0 - mu : 0.0);

    const auto [alpha_s_max, alpha_z_max] = get_max_step_sizes(
        s_dim, tau, workspace.vars.s, workspace.vars.z, ds, dz);

    const double k = std::max(max_neg_slope, settings.min_abs_merit_slope);
    const double rho =
        get_rho(settings.max_barrier_lagrangian_slope_for_zero_rho,
                settings.max_rho, bl_slope, penalty_multiplier_slope, k);

    const auto [m0_f, m0_s, m0_c, m0_g, m0_e, m0_aug, m0] =
        merit_function(settings, workspace, workspace.vars.s, workspace.vars.y,
                       workspace.vars.z, workspace.vars.e, mu, rho,
                       sq_constraint_violation_norm);

    const double merit_slope = bl_slope - rho * penalty_multiplier_slope;

    bool ls_succeeded = false;
    double alpha = 1.0;
    do {
      for (int i = 0; i < x_dim; ++i) {
        workspace.next_vars.x[i] = workspace.vars.x[i] + alpha * dx[i];
      }

      for (int i = 0; i < y_dim; ++i) {
        workspace.next_vars.y[i] = workspace.vars.y[i] + alpha * dy[i];
      }

      const double alpha_s = std::min(alpha, alpha_s_max);
      const double alpha_z = std::min(alpha, alpha_z_max);
      for (int i = 0; i < s_dim; ++i) {
        workspace.next_vars.s[i] = workspace.vars.s[i] + alpha_s * ds[i];
        workspace.next_vars.z[i] = workspace.vars.z[i] + alpha_z * dz[i];
      }

      if (settings.enable_elastics) {
        for (int i = 0; i < s_dim; ++i) {
          workspace.next_vars.e[i] = workspace.vars.e[i] + alpha * de[i];
        }
      }

      ModelCallbackInput mci{
          .x = workspace.next_vars.x,
      };
      input.model_callback(mci, &workspace.model_callback_output);

      add(workspace.model_callback_output->g, workspace.next_vars.s, s_dim,
          workspace.miscellaneous_workspace.g_plus_s);

      if (settings.enable_elastics) {
        add(workspace.miscellaneous_workspace.g_plus_s, workspace.next_vars.e,
            s_dim, workspace.miscellaneous_workspace.g_plus_s_plus_e);
      } else {
        std::copy(workspace.miscellaneous_workspace.g_plus_s,
                  workspace.miscellaneous_workspace.g_plus_s + s_dim,
                  workspace.miscellaneous_workspace.g_plus_s_plus_e);
      }

      const double next_sq_constraint_violation_norm =
          squared_norm(workspace.model_callback_output->c, y_dim) +
          squared_norm(workspace.miscellaneous_workspace.g_plus_s_plus_e,
                       s_dim);

      // TODO(joao): cache the barrier Lagrangian into the next cycle!
      const auto [m_f, m_s, m_c, m_g, m_e, m_aug, m] = merit_function(
          settings, workspace, workspace.next_vars.s, workspace.next_vars.y,
          workspace.next_vars.z, workspace.next_vars.e, mu, rho,
          next_sq_constraint_violation_norm);

      const double dm_f = m_f - m0_f;
      const double dm_s = m_s - m0_s;
      const double dm_c = m_c - m0_c;
      const double dm_g = m_g - m0_g;
      const double dm_e = m_e - m0_e;
      const double dm_aug = m_aug - m0_aug;

      const double merit_delta = dm_f + dm_s + dm_c + dm_g + dm_e + dm_aug;

      if (merit_slope > settings.min_merit_slope_to_skip_line_search ||
          merit_delta < settings.armijo_factor * merit_slope * alpha) {
        ls_succeeded = true;
        break;
      }

      alpha *= settings.line_search_factor;
    } while (alpha > settings.line_search_min_step_size);

    if (alpha <= settings.line_search_min_step_size) {
      alpha /= settings.line_search_factor;
    }

    std::swap(workspace.vars.x, workspace.next_vars.x);
    std::swap(workspace.vars.s, workspace.next_vars.s);
    std::swap(workspace.vars.y, workspace.next_vars.y);
    std::swap(workspace.vars.z, workspace.next_vars.z);
    if (settings.enable_elastics) {
      std::swap(workspace.vars.e, workspace.next_vars.e);
    }

    if (settings.print_logs) {
      const double de_norm = settings.enable_elastics ? norm(de, s_dim) : -1.0;
      std::cout << fmt::format(
                       // clang-format off
                       "{:^+10} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g}",
                       // clang-format on
                       iteration, alpha, m0, f0, std::sqrt(ctc),
                       std::sqrt(gsetgse), bl_slope, merit_slope, alpha_s_max,
                       alpha_z_max, norm(dx, x_dim), norm(ds, s_dim),
                       norm(dy, y_dim), norm(dz, s_dim), de_norm, mu, rho,
                       lin_sys_error, kkt_error)
                << std::endl;
    }

    if (settings.enable_line_search_failures && !ls_succeeded) {
      output.exit_status = Status::LINE_SEARCH_FAILURE;
      output.num_iterations = iteration;
      return;
    }
  }

  output.exit_status = Status::ITERATION_LIMIT;
  output.num_iterations = settings.max_iterations;
}

void ModelCallbackOutput::reserve(int x_dim, int s_dim, int y_dim,
                                  int upper_hessian_f_nnz, int jacobian_c_nnz,
                                  int jacobian_g_nnz,
                                  bool is_jacobian_c_transposed,
                                  bool is_jacobian_g_transposed) {
  gradient_f = new double[x_dim];
  upper_hessian_f.reserve(x_dim, upper_hessian_f_nnz);
  c = new double[y_dim];
  if (is_jacobian_c_transposed) {
    jacobian_c.reserve(y_dim, jacobian_c_nnz);
  } else {
    jacobian_c.reserve(x_dim, jacobian_c_nnz);
  }
  g = new double[s_dim];
  if (is_jacobian_g_transposed) {
    jacobian_g.reserve(s_dim, jacobian_g_nnz);
  } else {
    jacobian_g.reserve(x_dim, jacobian_g_nnz);
  }
}

void ModelCallbackOutput::free() {
  delete[] gradient_f;
  upper_hessian_f.free();
  delete[] c;
  jacobian_c.free();
  delete[] g;
  jacobian_g.free();
}

auto ModelCallbackOutput::mem_assign(int x_dim, int s_dim, int y_dim,
                                     int upper_hessian_f_nnz,
                                     int jacobian_c_nnz, int jacobian_g_nnz,
                                     bool is_jacobian_c_transposed,
                                     bool is_jacobian_g_transposed,
                                     unsigned char *mem_ptr) -> int {
  int cum_size = 0;
  gradient_f = reinterpret_cast<decltype(gradient_f)>(mem_ptr + cum_size);
  cum_size += x_dim * sizeof(double);

  cum_size += upper_hessian_f.mem_assign(x_dim, upper_hessian_f_nnz,
                                         mem_ptr + cum_size);

  c = reinterpret_cast<decltype(c)>(mem_ptr + cum_size);
  cum_size += y_dim * sizeof(double);

  if (is_jacobian_c_transposed) {
    cum_size +=
        jacobian_c.mem_assign(y_dim, jacobian_c_nnz, mem_ptr + cum_size);
  } else {
    cum_size +=
        jacobian_c.mem_assign(x_dim, jacobian_c_nnz, mem_ptr + cum_size);
  }

  g = reinterpret_cast<decltype(g)>(mem_ptr + cum_size);
  cum_size += s_dim * sizeof(double);

  if (is_jacobian_g_transposed) {
    cum_size +=
        jacobian_g.mem_assign(s_dim, jacobian_g_nnz, mem_ptr + cum_size);
  } else {
    cum_size +=
        jacobian_g.mem_assign(x_dim, jacobian_g_nnz, mem_ptr + cum_size);
  }

  return cum_size;
}

void VariablesWorkspace::reserve(int x_dim, int s_dim, int y_dim) {
  x = new double[x_dim];
  s = new double[s_dim];
  y = new double[y_dim];
  z = new double[s_dim];
  e = new double[s_dim];
}

void VariablesWorkspace::free() {
  delete[] x;
  delete[] s;
  delete[] y;
  delete[] z;
  delete[] e;
}

auto VariablesWorkspace::mem_assign(int x_dim, int s_dim, int y_dim,
                                    unsigned char *mem_ptr) -> int {
  int cum_size = 0;

  x = reinterpret_cast<decltype(x)>(mem_ptr + cum_size);
  cum_size += x_dim * sizeof(double);

  s = reinterpret_cast<decltype(s)>(mem_ptr + cum_size);
  cum_size += s_dim * sizeof(double);

  y = reinterpret_cast<decltype(y)>(mem_ptr + cum_size);
  cum_size += y_dim * sizeof(double);

  z = reinterpret_cast<decltype(z)>(mem_ptr + cum_size);
  cum_size += s_dim * sizeof(double);

  e = reinterpret_cast<decltype(e)>(mem_ptr + cum_size);
  cum_size += s_dim * sizeof(double);

  return cum_size;
}

void MiscellaneousWorkspace::reserve(int s_dim) {
  g_plus_s = new double[s_dim];
  g_plus_s_plus_e = new double[s_dim];
}

void MiscellaneousWorkspace::free() {
  delete[] g_plus_s;
  delete[] g_plus_s_plus_e;
}

auto MiscellaneousWorkspace::mem_assign(int s_dim, unsigned char *mem_ptr)
    -> int {
  int cum_size = 0;

  g_plus_s = reinterpret_cast<decltype(g_plus_s)>(mem_ptr + cum_size);
  cum_size += s_dim * sizeof(double);

  g_plus_s_plus_e =
      reinterpret_cast<decltype(g_plus_s_plus_e)>(mem_ptr + cum_size);
  cum_size += s_dim * sizeof(double);

  return cum_size;
}

void Workspace::reserve(int x_dim, int s_dim, int y_dim) {
  vars.reserve(x_dim, s_dim, y_dim);
  delta_vars.reserve(x_dim, s_dim, y_dim);
  next_vars.reserve(x_dim, s_dim, y_dim);
  nrhs.reserve(x_dim, s_dim, y_dim);
  miscellaneous_workspace.reserve(s_dim);
}

void Workspace::free() {
  vars.free();
  delta_vars.free();
  next_vars.free();
  nrhs.free();
  miscellaneous_workspace.free();
}

auto Workspace::mem_assign(int x_dim, int s_dim, int y_dim,
                           unsigned char *mem_ptr) -> int {
  int cum_size = 0;

  cum_size += vars.mem_assign(x_dim, s_dim, y_dim, mem_ptr + cum_size);
  cum_size += delta_vars.mem_assign(x_dim, s_dim, y_dim, mem_ptr + cum_size);
  cum_size += next_vars.mem_assign(x_dim, s_dim, y_dim, mem_ptr + cum_size);
  cum_size += nrhs.mem_assign(x_dim, s_dim, y_dim, mem_ptr + cum_size);
  cum_size += miscellaneous_workspace.mem_assign(s_dim, mem_ptr + cum_size);

  return cum_size;
}

auto operator<<(std::ostream &os, Status const &status) -> std::ostream & {
  switch (status) {
  case Status::SOLVED:
    os << "SOLVED";
    break;
  case Status::ITERATION_LIMIT:
    os << "ITERATION_LIMIT";
    break;
  case Status::LINE_SEARCH_FAILURE:
    os << "LINE_SEARCH_FAILURE";
    break;
  }
  return os;
}

} // namespace sip
