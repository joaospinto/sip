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

auto get_alpha_s_max(const int s_dim, const double tau, const double *s,
                     const double *ds) -> double {
  // s + alpha_s_max * ds >= (1 - tau) * s

  double alpha_s_max = 1.0;

  for (int i = 0; i < s_dim; ++i) {
    if (ds[i] < 0.0) {
      alpha_s_max = std::min(alpha_s_max, tau * s[i] / std::max(-ds[i], 1e-12));
    }
  }

  return alpha_s_max;
}

auto augmented_barrier_lagrangian_slope(const Settings &settings,
                                        const Workspace &workspace,
                                        const double *dx, const double *ds,
                                        const double *dy, const double *dz,
                                        const double *de) -> double {
  const auto &mco = *workspace.model_callback_output;
  const int x_dim = mco.upper_hessian_lagrangian.rows;
  const int s_dim = get_s_dim(mco.jacobian_g);
  const int y_dim = get_y_dim(mco.jacobian_c);
  const double x_slope = dot(workspace.nrhs.x, dx, x_dim);
  const double s_slope = dot(workspace.nrhs.s, ds, s_dim);
  const double y_slope = dot(workspace.nrhs.y, dy, y_dim);
  const double z_slope = dot(workspace.nrhs.z, dz, s_dim);
  const double e_slope =
      settings.enable_elastics ? dot(workspace.nrhs.e, de, s_dim) : 0.0;
  const double abl_slope = x_slope + s_slope + y_slope + z_slope + e_slope;
  return abl_slope;
}

auto merit_function(const Settings &settings, const Workspace &workspace,
                    const double *s, const double *y, const double *z,
                    const double *e, const double mu, const double eta,
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
  const double aug_term = 0.5 * eta * sq_constraint_violation_norm;
  const double merit = barrier_lagrangian + aug_term;
  return std::make_tuple(mco.f, s_term, c_term, g_term, e_term, aug_term,
                         merit);
}

auto compute_search_direction(const Input &input, const Settings &settings,
                              const double eta, const double mu,
                              Workspace &workspace)
    -> std::tuple<const double *, const double *, const double *,
                  const double *, const double *, double, double> {
  const double rho = settings.enable_elastics
                         ? settings.elastic_var_cost_coeff
                         : std::numeric_limits<double>::signaling_NaN();
  const auto &mco = *workspace.model_callback_output;

  const int x_dim = mco.upper_hessian_lagrangian.cols;
  const int s_dim = get_s_dim(mco.jacobian_g);
  const int y_dim = get_y_dim(mco.jacobian_c);
  const int dim_3x3 = x_dim + s_dim + y_dim;

  double kkt_error;
  double lin_sys_error = std::numeric_limits<double>::signaling_NaN();

  const double *s = workspace.vars.s;
  const double *y = workspace.vars.y;
  const double *z = workspace.vars.z;
  const double *e = workspace.vars.e;

  const double *H_data = mco.upper_hessian_lagrangian.data;
  const double *C_data = mco.jacobian_c.data;
  const double *G_data = mco.jacobian_g.data;
  const double *grad_f = mco.gradient_f;

  const double *c = mco.c;
  const double *gpspe = workspace.miscellaneous_workspace.g_plus_s_plus_e;

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

  double *w = workspace.csd_workspace.w;
  double *y_tilde = workspace.csd_workspace.y_tilde;
  double *z_tilde = workspace.csd_workspace.z_tilde;
  double *LT_data = workspace.csd_workspace.LT_data;
  double *D_diag = workspace.csd_workspace.D_diag;
  double *b = workspace.csd_workspace.rhs_block_3x3;
  double *v = workspace.csd_workspace.sol_block_3x3;

  for (int i = 0; i < y_dim; ++i) {
    y_tilde[i] = y[i] + eta * c[i];
  }

  for (int i = 0; i < s_dim; ++i) {
    w[i] = s[i] / z[i];
    z_tilde[i] = z[i] + eta * gpspe[i];
  }

  constexpr double r1 = 0.0;
  const double eta_inv = 1.0 / eta;
  const double r3p = settings.enable_elastics ? eta_inv + 1.0 / rho : eta_inv;

  input.ldlt_factor(H_data, C_data, G_data, w, r1, eta_inv, r3p, LT_data,
                    D_diag);

  double *bx = b;
  double *by = bx + x_dim;
  double *bz = by + y_dim;

  for (int i = 0; i < x_dim; ++i) {
    bx[i] = grad_f[i];
  }

  input.add_CTx_to_y(C_data, y_tilde, bx);
  input.add_GTx_to_y(G_data, z_tilde, bx);

  std::copy(bx, bx + x_dim, rx);

  std::fill(by, by + y_dim, 0.0);
  std::fill(ry, ry + y_dim, 0.0);

  if (settings.enable_elastics) {
    for (int i = 0; i < s_dim; ++i) {
      rs[i] = z_tilde[i] - mu / s[i];
      re[i] = rho * e[i] + z_tilde[i];
      rz[i] = 0.0;
      bz[i] = -re[i] / rho - w[i] * rs[i];
    }
  } else {
    for (int i = 0; i < s_dim; ++i) {
      rs[i] = z_tilde[i] - mu / s[i];
      rz[i] = 0.0;
      bz[i] = -w[i] * rs[i];
    }
  }

  for (int i = 0; i < dim_3x3; ++i) {
    b[i] = -b[i];
  }

  input.ldlt_solve(LT_data, D_diag, b, v);

  {
    auto ptr = v;
    std::copy(ptr, ptr + x_dim, dx);
    ptr += x_dim;
    std::copy(ptr, ptr + y_dim, dy);
    ptr += y_dim;
    std::copy(ptr, ptr + s_dim, dz);
    ptr += s_dim;
  }

  for (int i = 0; i < s_dim; ++i) {
    ds[i] = -w[i] * (dz[i] + rs[i]);
  }

  if (settings.enable_elastics) {
    for (int i = 0; i < s_dim; ++i) {
      de[i] = -(dz[i] + re[i]) / rho;
    }
  }

  if (settings.print_logs) {
    double *residual = workspace.csd_workspace.residual;
    double *res_x = residual;
    double *res_s = res_x + x_dim;
    double *res_y = res_s + s_dim;
    double *res_z = res_y + y_dim;

    for (int i = 0; i < x_dim; ++i) {
      res_x[i] = -bx[i];
    }

    for (int i = 0; i < y_dim; ++i) {
      res_y[i] = -by[i];
    }

    for (int i = 0; i < s_dim; ++i) {
      res_z[i] = -bz[i];
    }

    input.add_Kx_to_y(H_data, C_data, G_data, w, r1, eta_inv, r3p, dx, dy, dz,
                      res_x, res_y, res_z);

    for (int i = 0; i < s_dim; ++i) {
      res_s[i] = ds[i] / w[i] + dz[i] + rs[i];
    }

    if (settings.enable_elastics) {
      double *res_e = res_z + s_dim;
      for (int i = 0; i < s_dim; ++i) {
        res_e[i] = rho * de[i] + dz[i] + re[i];
      }
    }

    lin_sys_error = 0.0;

    int full_dim = x_dim + y_dim + s_dim + s_dim;
    if (settings.enable_elastics) {
      full_dim += s_dim;
    }
    for (int i = 0; i < full_dim; ++i) {
      lin_sys_error = std::max(lin_sys_error, std::fabs(residual[i]));
    }
  }

  kkt_error = 0.0;

  for (int i = 0; i < x_dim; ++i) {
    kkt_error = std::max(kkt_error, std::fabs(bx[i]));
  }
  for (int i = 0; i < y_dim; ++i) {
    kkt_error = std::max(kkt_error, std::fabs(c[i]));
  }
  for (int i = 0; i < s_dim; ++i) {
    kkt_error = std::max(
        kkt_error,
        std::fabs(workspace.miscellaneous_workspace.g_plus_s_plus_e[i]));
  }

  return std::make_tuple(dx, ds, dy, dz, de, kkt_error, lin_sys_error);
}

auto check_settings(const Settings &settings) -> bool {
  return !settings.enable_elastics || settings.elastic_var_cost_coeff > 0.0;
}

auto solve(const Input &input, const Settings &settings, Workspace &workspace,
           Output &output) -> void {
  {
    ModelCallbackInput mci{
        .x = workspace.vars.x,
        .y = workspace.vars.y,
        .z = workspace.vars.z,
    };

    input.model_callback(mci, &workspace.model_callback_output);
  }

  if (!check_settings(settings)) {
    if (settings.assert_checks_pass) {
      assert(false && "check_settings returned false.");
    } else {
      output.exit_status = Status::FAILED_CHECK;
      output.num_iterations = 0;
      return;
    }
  }

  const int x_dim =
      workspace.model_callback_output->upper_hessian_lagrangian.cols;
  const int s_dim = get_s_dim(workspace.model_callback_output->jacobian_g);
  const int y_dim = get_y_dim(workspace.model_callback_output->jacobian_c);

  for (int i = 0; i < s_dim; ++i) {
    if (workspace.vars.s[i] <= 0.0) {
      if (settings.assert_checks_pass) {
        assert(false && "workspace.vars.s[i] <= 0.0.");
      } else {
        output.exit_status = Status::FAILED_CHECK;
        output.num_iterations = 0;
        return;
      }
    }
    if (workspace.vars.z[i] <= 0.0) {
      if (settings.assert_checks_pass) {
        assert(false && "workspace.vars.z[i] <= 0.0.");
      } else {
        output.exit_status = Status::FAILED_CHECK;
        output.num_iterations = 0;
        return;
      }
    }
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

  double mu = settings.initial_mu;
  double eta = settings.initial_penalty_parameter;

  if (settings.print_logs) {
    std::cout
        << fmt::format(
               // clang-format off
                     "{:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10}",
               // clang-format on
               "iteration", "alpha", "merit", "f", "|c|", "|g+s+e|", "m_slope",
               "alpha_s_m", "|dx|", "|ds|", "|dy|", "|dz|", "|de|", "mu", "eta",
               "linsys_res", "kkt_error")
        << std::endl;
  }

  for (int iteration = 0; iteration < settings.max_iterations; ++iteration) {
    mu = std::max(mu * settings.mu_update_factor, settings.mu_min);

    const double f0 = workspace.model_callback_output->f;

    const double ctc = squared_norm(workspace.model_callback_output->c, y_dim);
    const double gsetgse =
        squared_norm(workspace.miscellaneous_workspace.g_plus_s_plus_e, s_dim);

    const double sq_constraint_violation_norm = ctc + gsetgse;

    const auto [dx, ds, dy, dz, de, kkt_error, lin_sys_error] =
        compute_search_direction(input, settings, eta, mu, workspace);

    const double merit_slope = augmented_barrier_lagrangian_slope(
        settings, workspace, dx, ds, dy, dz, de);

    if (kkt_error < settings.max_kkt_violation) {
      if (settings.print_logs) {
        const double de_norm =
            settings.enable_elastics ? norm(de, s_dim) : -1.0;
        std::cout
            << fmt::format(
                   // clang-format off
                         "{:^+10} {:^10} {:^10} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^10} {:^10} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^10} {:^+10.4g} {:^+10.4g}",
                   // clang-format on
                   iteration, "", "", workspace.model_callback_output->f,
                   std::sqrt(ctc), std::sqrt(gsetgse), "", "", norm(dx, x_dim),
                   norm(ds, s_dim), norm(dy, y_dim), norm(dz, s_dim), de_norm,
                   mu, "", lin_sys_error, kkt_error)
            << std::endl;
      }

      output.exit_status = Status::SOLVED;
      output.num_iterations = iteration;
      return;
    }

    const double tau = std::max(settings.tau_min, mu > 0.0 ? 1.0 - mu : 0.0);

    const auto alpha_s_max = get_alpha_s_max(s_dim, tau, workspace.vars.s, ds);

    const auto [m0_f, m0_s, m0_c, m0_g, m0_e, m0_aug, m0] =
        merit_function(settings, workspace, workspace.vars.s, workspace.vars.y,
                       workspace.vars.z, workspace.vars.e, mu, eta,
                       sq_constraint_violation_norm);

    bool ls_succeeded = false;
    double alpha = 1.0;
    double constraint_violation_ratio =
        std::numeric_limits<double>::signaling_NaN();
    do {
      for (int i = 0; i < x_dim; ++i) {
        workspace.next_vars.x[i] = workspace.vars.x[i] + alpha * dx[i];
      }

      for (int i = 0; i < y_dim; ++i) {
        workspace.next_vars.y[i] =
            workspace.csd_workspace.y_tilde[i] + alpha * dy[i];
      }

      for (int i = 0; i < s_dim; ++i) {
        workspace.next_vars.s[i] = std::max(workspace.vars.s[i] + alpha * ds[i],
                                            (1.0 - tau) * workspace.vars.s[i]);
        workspace.next_vars.z[i] =
            std::max(workspace.csd_workspace.z_tilde[i] + alpha * dz[i],
                     (1.0 - tau) * workspace.vars.z[i]);
      }

      if (settings.enable_elastics) {
        for (int i = 0; i < s_dim; ++i) {
          workspace.next_vars.e[i] = workspace.vars.e[i] + alpha * de[i];
        }
      }

      ModelCallbackInput mci{
          .x = workspace.next_vars.x,
          .y = workspace.next_vars.y,
          .z = workspace.next_vars.z,
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

      const auto [m_f, m_s, m_c, m_g, m_e, m_aug, m] = merit_function(
          settings, workspace, workspace.next_vars.s, workspace.vars.y,
          workspace.vars.z, workspace.next_vars.e, mu, eta,
          next_sq_constraint_violation_norm);

      const double dm_f = m_f - m0_f;
      const double dm_s = m_s - m0_s;
      const double dm_c = m_c - m0_c;
      const double dm_g = m_g - m0_g;
      const double dm_e = m_e - m0_e;
      const double dm_aug = m_aug - m0_aug;

      const double merit_delta = dm_f + dm_s + dm_c + dm_g + dm_e + dm_aug;

      constraint_violation_ratio =
          next_sq_constraint_violation_norm /
          std::max(sq_constraint_violation_norm, 1e-12);

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

    std::swap(workspace.vars, workspace.next_vars);

    if (constraint_violation_ratio >
        settings.min_acceptable_constraint_violation_ratio) {
      eta = std::min(eta * settings.penalty_parameter_increase_factor,
                     settings.max_penalty_parameter);
    } else {
      eta *= settings.penalty_parameter_decrease_factor;
    }

    if (settings.print_logs) {
      const double de_norm = settings.enable_elastics ? norm(de, s_dim) : -1.0;
      std::cout << fmt::format(
                       // clang-format off
                       "{:^+10} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g}",
                       // clang-format on
                       iteration, alpha, m0, f0, std::sqrt(ctc),
                       std::sqrt(gsetgse), merit_slope, alpha_s_max,
                       norm(dx, x_dim), norm(ds, s_dim), norm(dy, y_dim),
                       norm(dz, s_dim), de_norm, mu, eta, lin_sys_error,
                       kkt_error)
                << std::endl;
    }

    if (settings.enable_line_search_failures && !ls_succeeded) {
      output.exit_status = Status::LINE_SEARCH_FAILURE;
      output.num_iterations = iteration;
      return;
    }

    if (input.timeout_callback()) {
      output.exit_status = Status::TIMEOUT;
      output.num_iterations = iteration;
      return;
    }
  }

  output.exit_status = Status::ITERATION_LIMIT;
  output.num_iterations = settings.max_iterations;
}

void ModelCallbackOutput::reserve(int x_dim, int s_dim, int y_dim,
                                  int upper_hessian_lagrangian_nnz,
                                  int jacobian_c_nnz, int jacobian_g_nnz,
                                  bool is_jacobian_c_transposed,
                                  bool is_jacobian_g_transposed) {
  gradient_f = new double[x_dim];
  upper_hessian_lagrangian.reserve(x_dim, upper_hessian_lagrangian_nnz);
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
  upper_hessian_lagrangian.free();
  delete[] c;
  jacobian_c.free();
  delete[] g;
  jacobian_g.free();
}

auto ModelCallbackOutput::mem_assign(int x_dim, int s_dim, int y_dim,
                                     int upper_hessian_lagrangian_nnz,
                                     int jacobian_c_nnz, int jacobian_g_nnz,
                                     bool is_jacobian_c_transposed,
                                     bool is_jacobian_g_transposed,
                                     unsigned char *mem_ptr) -> int {
  int cum_size = 0;
  gradient_f = reinterpret_cast<decltype(gradient_f)>(mem_ptr + cum_size);
  cum_size += x_dim * sizeof(double);

  cum_size += upper_hessian_lagrangian.mem_assign(
      x_dim, upper_hessian_lagrangian_nnz, mem_ptr + cum_size);

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

void ComputeSearchDirectionWorkspace::reserve(int s_dim, int y_dim, int kkt_dim,
                                              int full_dim, int L_nnz) {
  w = new double[s_dim];
  y_tilde = new double[y_dim];
  z_tilde = new double[s_dim];
  LT_data = new double[L_nnz];
  D_diag = new double[kkt_dim];
  rhs_block_3x3 = new double[kkt_dim];
  sol_block_3x3 = new double[kkt_dim];
  residual = new double[full_dim];
}

void ComputeSearchDirectionWorkspace::free() {
  delete[] w;
  delete[] y_tilde;
  delete[] z_tilde;
  delete[] LT_data;
  delete[] D_diag;
  delete[] rhs_block_3x3;
  delete[] sol_block_3x3;
  delete[] residual;
}

auto ComputeSearchDirectionWorkspace::mem_assign(int s_dim, int y_dim,
                                                 int kkt_dim, int full_dim,
                                                 int L_nnz,
                                                 unsigned char *mem_ptr)
    -> int {
  int cum_size = 0;

  w = reinterpret_cast<decltype(w)>(mem_ptr + cum_size);
  cum_size += s_dim * sizeof(double);
  y_tilde = reinterpret_cast<decltype(y_tilde)>(mem_ptr + cum_size);
  cum_size += y_dim * sizeof(double);
  z_tilde = reinterpret_cast<decltype(z_tilde)>(mem_ptr + cum_size);
  cum_size += s_dim * sizeof(double);
  LT_data = reinterpret_cast<decltype(LT_data)>(mem_ptr + cum_size);
  cum_size += L_nnz * sizeof(double);
  D_diag = reinterpret_cast<decltype(D_diag)>(mem_ptr + cum_size);
  cum_size += kkt_dim * sizeof(double);
  rhs_block_3x3 = reinterpret_cast<decltype(rhs_block_3x3)>(mem_ptr + cum_size);
  cum_size += kkt_dim * sizeof(double);
  sol_block_3x3 = reinterpret_cast<decltype(sol_block_3x3)>(mem_ptr + cum_size);
  cum_size += kkt_dim * sizeof(double);
  residual = reinterpret_cast<decltype(residual)>(mem_ptr + cum_size);
  cum_size += full_dim * sizeof(double);

  return cum_size;
}

void Workspace::reserve(int x_dim, int s_dim, int y_dim, int L_nnz) {
  vars.reserve(x_dim, s_dim, y_dim);
  delta_vars.reserve(x_dim, s_dim, y_dim);
  next_vars.reserve(x_dim, s_dim, y_dim);
  nrhs.reserve(x_dim, s_dim, y_dim);
  miscellaneous_workspace.reserve(s_dim);
  const int kkt_dim = x_dim + s_dim + y_dim;
  const int full_dim = kkt_dim + s_dim + s_dim;
  csd_workspace.reserve(s_dim, y_dim, kkt_dim, full_dim, L_nnz);
}

void Workspace::free() {
  vars.free();
  delta_vars.free();
  next_vars.free();
  nrhs.free();
  miscellaneous_workspace.free();
  csd_workspace.free();
}

auto Workspace::mem_assign(int x_dim, int s_dim, int y_dim, int L_nnz,
                           unsigned char *mem_ptr) -> int {
  int cum_size = 0;

  cum_size += vars.mem_assign(x_dim, s_dim, y_dim, mem_ptr + cum_size);
  cum_size += delta_vars.mem_assign(x_dim, s_dim, y_dim, mem_ptr + cum_size);
  cum_size += next_vars.mem_assign(x_dim, s_dim, y_dim, mem_ptr + cum_size);
  cum_size += nrhs.mem_assign(x_dim, s_dim, y_dim, mem_ptr + cum_size);
  cum_size += miscellaneous_workspace.mem_assign(s_dim, mem_ptr + cum_size);
  const int kkt_dim = x_dim + s_dim + y_dim;
  const int full_dim = kkt_dim + s_dim + s_dim;
  cum_size += csd_workspace.mem_assign(s_dim, y_dim, kkt_dim, full_dim, L_nnz,
                                       mem_ptr + cum_size);

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
  case Status::TIMEOUT:
    os << "TIMEOUT";
    break;
  case Status::FAILED_CHECK:
    os << "FAILED_CHECK";
    break;
  }
  return os;
}

} // namespace sip
