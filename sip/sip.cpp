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
      alpha_s_max = std::min(alpha_s_max, tau * s[i] / std::max(-ds[i], 1e-12));
    }
    if (dz[i] < 0.0) {
      alpha_z_max = std::min(alpha_z_max, tau * z[i] / std::max(-dz[i], 1e-12));
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

auto get_rho(const Settings &settings, const Workspace &workspace,
             const double *s, const double *e, const double *dx,
             const double *ds, const double *de, const double mu) -> double {
  // D(merit_function; dx, ds) = D(f; dx) - mu (ds / s) - rho * ||c(x)|| - rho *
  // ||g(x) + s || rho > (D(f; dx) + k) - mu (ds / s) + k) / (|| (c(x) || +
  // || g(x) + s) || iff D(merit_function; dx) < -k.
  const auto &mco = *workspace.model_callback_output;
  const int x_dim = mco.upper_hessian_f.rows;
  const int s_dim = get_s_dim(mco.jacobian_g);
  const int y_dim = get_y_dim(mco.jacobian_c);
  const double f_slope = dot(mco.gradient_f, dx, x_dim);
  const double barrier_slope = -mu * x_dot_y_inverse(ds, s, s_dim);
  const double elastic_slope =
      settings.enable_elastics
          ? settings.elastic_var_cost_coeff * dot(de, e, s_dim)
          : 0.0;
  const double obj_slope = f_slope + barrier_slope + elastic_slope;
  const double d =
      norm(mco.c, y_dim) +
      norm(workspace.miscellaneous_workspace.g_plus_s_plus_e, s_dim);
  const double k = std::max(d, 2.0 * std::fabs(obj_slope));
  return std::min((obj_slope + k) / d, 1e9);
}

auto merit_function(const Settings &settings, const Workspace &workspace,
                    const double *s, const double *e, const double mu,
                    const double rho) -> double {
  const auto &mco = *workspace.model_callback_output;
  const int s_dim = get_s_dim(mco.jacobian_g);
  const int y_dim = get_y_dim(mco.jacobian_c);
  const double e_term =
      settings.enable_elastics
          ? 0.5 * settings.elastic_var_cost_coeff * squared_norm(e, s_dim)
          : 0.0;
  const double s_term = -mu * sum_of_logs(s, s_dim);
  return mco.f + e_term + s_term + rho * norm(mco.c, y_dim) +
         rho * norm(workspace.miscellaneous_workspace.g_plus_s_plus_e, s_dim);
}

auto merit_function_slope(const Settings &settings, const Workspace &workspace,
                          const double *s, const double *e, const double *dx,
                          const double *ds, const double *de, const double mu,
                          const double rho) {
  // TODO(joao): eventually remove repeated computation across:
  // 1. merit_function
  // 2. merit_function_slope
  // 3. get_rho
  const auto &mco = *workspace.model_callback_output;
  const int x_dim = mco.upper_hessian_f.rows;
  const int s_dim = get_s_dim(mco.jacobian_g);
  const int y_dim = get_y_dim(mco.jacobian_c);
  const double f_slope = dot(mco.gradient_f, dx, x_dim);
  const double barrier_slope = -mu * x_dot_y_inverse(ds, s, s_dim);
  const double elastic_slope =
      settings.enable_elastics
          ? settings.elastic_var_cost_coeff * dot(de, e, s_dim)
          : 0.0;
  const double obj_slope = f_slope + barrier_slope + elastic_slope;
  return obj_slope - rho * norm(mco.c, y_dim) -
         rho * norm(workspace.miscellaneous_workspace.g_plus_s_plus_e, s_dim);
}

auto compute_search_direction(const Input &input, const Settings &settings,
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
  double *dx = workspace.vars.dx;
  double *ds = workspace.vars.ds;
  double *dy = workspace.vars.dy;
  double *dz = workspace.vars.dz;
  double *de = workspace.vars.de;

  input.lin_sys_solver(mco.c, mco.g, mco.gradient_f, mco.upper_hessian_f.data,
                       mco.jacobian_c.data, mco.jacobian_g.data,
                       workspace.vars.s, workspace.vars.y, workspace.vars.z,
                       workspace.vars.e, mu, p, gamma_x, settings.gamma_y,
                       settings.gamma_z, dx, ds, dy, dz, de, kkt_error,
                       lin_sys_error);

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
    std::cout
        << fmt::format(
               // clang-format off
                     "{:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10}",
               // clang-format on
               "iteration", "alpha", "merit", "f", "|c|", "|g+s+e|", "m_slope",
               "alpha_s_m", "alpha_z_m", "|dx|", "|ds|", "|dy|", "|dz|", "|de|",
               "mu", "rho", "linsys_res", "kkt_error")
        << std::endl;
  }

  for (int iteration = 0; iteration < settings.max_iterations; ++iteration) {
    const double mu =
        std::max(adaptive_mu(s_dim, workspace.vars.s, workspace.vars.z),
                 settings.mu_min);

    const auto [dx, ds, dy, dz, de, kkt_error, lin_sys_error] =
        compute_search_direction(input, settings, mu, workspace);

    if (kkt_error < settings.max_kkt_violation) {
      output.exit_status = Status::SOLVED;
      output.num_iterations = iteration;
      return;
    }

    const double tau = std::max(settings.tau_min, mu > 0.0 ? 1.0 - mu : 0.0);

    const auto [alpha_s_max, alpha_z_max] = get_max_step_sizes(
        s_dim, tau, workspace.vars.s, workspace.vars.z, ds, dz);

    const double rho = get_rho(settings, workspace, workspace.vars.s,
                               workspace.vars.e, dx, ds, de, mu);

    const double merit = merit_function(settings, workspace, workspace.vars.s,
                                        workspace.vars.e, mu, rho);

    const double merit_slope =
        merit_function_slope(settings, workspace, workspace.vars.s,
                             workspace.vars.e, dx, ds, de, mu, rho);

    bool ls_succeeded = false;
    double new_merit = std::numeric_limits<double>::signaling_NaN();
    double alpha = alpha_s_max;
    do {
      for (int i = 0; i < x_dim; ++i) {
        workspace.vars.next_x[i] = workspace.vars.x[i] + alpha * dx[i];
      }

      for (int i = 0; i < s_dim; ++i) {
        workspace.vars.next_s[i] = workspace.vars.s[i] + alpha * ds[i];
      }

      if (settings.enable_elastics) {
        for (int i = 0; i < s_dim; ++i) {
          workspace.vars.next_e[i] = workspace.vars.e[i] + alpha * de[i];
        }
      }

      ModelCallbackInput mci{
          .x = workspace.vars.next_x,
      };
      input.model_callback(mci, &workspace.model_callback_output);

      add(workspace.model_callback_output->g, workspace.vars.next_s, s_dim,
          workspace.miscellaneous_workspace.g_plus_s);

      if (settings.enable_elastics) {
        add(workspace.miscellaneous_workspace.g_plus_s, workspace.vars.next_e,
            s_dim, workspace.miscellaneous_workspace.g_plus_s_plus_e);
      } else {
        std::copy(workspace.miscellaneous_workspace.g_plus_s,
                  workspace.miscellaneous_workspace.g_plus_s + s_dim,
                  workspace.miscellaneous_workspace.g_plus_s_plus_e);
      }

      // TODO(joao): cache (parts of) this into the next cycle!
      new_merit = merit_function(settings, workspace, workspace.vars.next_s,
                                 workspace.vars.next_e, mu, rho);

      if (new_merit - merit < settings.armijo_factor * merit_slope * alpha) {
        ls_succeeded = true;
        break;
      }

      alpha *= settings.line_search_factor;
    } while (alpha > settings.line_search_min_step_size);

    if (alpha <= settings.line_search_min_step_size) {
      alpha /= settings.line_search_factor;
    }

    std::swap(workspace.vars.x, workspace.vars.next_x);
    std::swap(workspace.vars.s, workspace.vars.next_s);
    if (settings.enable_elastics) {
      std::swap(workspace.vars.e, workspace.vars.next_e);
    }

    for (int i = 0; i < y_dim; ++i) {
      workspace.vars.y[i] += alpha_z_max * dy[i];
    }

    for (int i = 0; i < s_dim; ++i) {
      workspace.vars.z[i] += alpha_z_max * dz[i];
    }

    if (settings.print_logs) {
      const double de_norm = settings.enable_elastics ? norm(de, s_dim) : -1.0;
      std::cout << fmt::format(
                       // clang-format off
                       "{:^+10} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g}",
                       // clang-format on
                       iteration, alpha, new_merit,
                       workspace.model_callback_output->f,
                       norm(workspace.model_callback_output->c, y_dim),
                       norm(workspace.miscellaneous_workspace.g_plus_s_plus_e,
                            s_dim),
                       merit_slope, alpha_s_max, alpha_z_max, norm(dx, x_dim),
                       norm(ds, s_dim), norm(dy, y_dim), norm(dz, s_dim),
                       de_norm, mu, rho, lin_sys_error, kkt_error)
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
  next_x = new double[x_dim];
  next_s = new double[s_dim];
  next_e = new double[s_dim];
  dx = new double[x_dim];
  ds = new double[s_dim];
  dy = new double[y_dim];
  dz = new double[s_dim];
  de = new double[s_dim];
}

void VariablesWorkspace::free() {
  delete[] x;
  delete[] s;
  delete[] y;
  delete[] z;
  delete[] e;
  delete[] next_x;
  delete[] next_s;
  delete[] next_e;
  delete[] dx;
  delete[] ds;
  delete[] dy;
  delete[] dz;
  delete[] de;
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

  next_x = reinterpret_cast<decltype(next_x)>(mem_ptr + cum_size);
  cum_size += x_dim * sizeof(double);

  next_s = reinterpret_cast<decltype(next_s)>(mem_ptr + cum_size);
  cum_size += s_dim * sizeof(double);

  next_e = reinterpret_cast<decltype(next_e)>(mem_ptr + cum_size);
  cum_size += s_dim * sizeof(double);

  dx = reinterpret_cast<decltype(x)>(mem_ptr + cum_size);
  cum_size += x_dim * sizeof(double);

  ds = reinterpret_cast<decltype(s)>(mem_ptr + cum_size);
  cum_size += s_dim * sizeof(double);

  dy = reinterpret_cast<decltype(y)>(mem_ptr + cum_size);
  cum_size += y_dim * sizeof(double);

  dz = reinterpret_cast<decltype(z)>(mem_ptr + cum_size);
  cum_size += s_dim * sizeof(double);

  de = reinterpret_cast<decltype(e)>(mem_ptr + cum_size);
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
  miscellaneous_workspace.reserve(s_dim);
}

void Workspace::free() {
  vars.free();
  miscellaneous_workspace.free();
}

auto Workspace::mem_assign(int x_dim, int s_dim, int y_dim,
                           unsigned char *mem_ptr) -> int {
  int cum_size = 0;

  cum_size += vars.mem_assign(x_dim, s_dim, y_dim, mem_ptr + cum_size);
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
