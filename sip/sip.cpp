#include "sip/sip.hpp"
#include "sip/helpers.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <fmt/color.h>
#include <fmt/core.h>
#include <limits>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

namespace sip {

namespace {

auto get_s_dim(const SparseMatrix &jacobian_g) -> int {
  return jacobian_g.is_transposed ? jacobian_g.cols : jacobian_g.rows;
}

auto get_y_dim(const SparseMatrix &jacobian_c) -> int {
  return jacobian_c.is_transposed ? jacobian_c.cols : jacobian_c.rows;
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

void print_log_header() {
  fmt::print(fmt::emphasis::bold | fg(fmt::color::red),
             // clang-format off
             "{:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10}\n",
             // clang-format on
             "iteration", "alpha", "f", "|c|", "|g+s+e|", "merit", "|dx|",
             "|ds|", "|dy|", "|dz|", "|de|", "mu", "eta", "tau", "kkt_error");
}

void print_search_direction_log_header() {
  fmt::print(fmt::emphasis::bold | fg(fmt::color::green),
             // clang-format off
             "{:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10}\n",
             // clang-format on
             "", "linsys_res", "alpha_s_m", "m_slope", "m_slope_v2",
             "obs_slope", "m_sl_x", "obs_sl_x", "m_sl_s", "obs_sl_s", "m_sl_e",
             "obs_sl_e", "|nrhs_x|", "|nrhs_s|", "|nrhs_e|");
}

void print_line_search_log_header() {
  fmt::print(fmt::emphasis::bold | fg(fmt::color::yellow),
             // clang-format off
             "{:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10}\n",
             // clang-format on
             "", "ls_iteration", "alpha", "merit", "f", "|c|", "|g+s+e|", "dm",
             "dm/alpha", "dm[f]", "dm[s]", "dm[c]", "dm[g]", "dm[e]",
             "dm[aug]");
}

void print_derivative_check_log_header() {
  fmt::print(fmt::emphasis::bold | fg(fmt::color::orange),
             // clang-format off
             "{:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10}\n",
             // clang-format on
             "", "f/c/g", "out_index", "var_index", "rel_error", "abs_error",
             "est_slope", "theo_slope");
}

void update_next_primal_vars(const Input &input, const Settings &settings,
                             const double tau, Workspace &workspace,
                             const double alpha, const bool update_x,
                             const bool update_s, const bool update_e) {
  const int x_dim =
      workspace.model_callback_output->upper_hessian_lagrangian.rows;
  const int s_dim = get_s_dim(workspace.model_callback_output->jacobian_g);

  if (update_x) {
    for (int i = 0; i < x_dim; ++i) {
      workspace.next_vars.x[i] =
          workspace.vars.x[i] + alpha * workspace.delta_vars.x[i];
    }
  } else {
    std::copy_n(workspace.vars.x, x_dim, workspace.next_vars.x);
  }

  ModelCallbackInput mci{
      .x = workspace.next_vars.x,
      .y = workspace.vars.y,
      .z = workspace.vars.z,
      .new_x = true,
      .new_y = false,
      .new_z = false,
  };
  input.model_callback(mci, &workspace.model_callback_output);

  if (update_s) {
    for (int i = 0; i < s_dim; ++i) {
      workspace.next_vars.s[i] =
          std::max(workspace.vars.s[i] + alpha * workspace.delta_vars.s[i],
                   (1.0 - tau) * workspace.vars.s[i]);
    }
  } else {
    std::copy_n(workspace.vars.s, s_dim, workspace.next_vars.s);
  }

  add(workspace.model_callback_output->g, workspace.next_vars.s, s_dim,
      workspace.miscellaneous_workspace.g_plus_s);

  if (settings.enable_elastics) {
    if (update_e) {
      for (int i = 0; i < s_dim; ++i) {
        workspace.next_vars.e[i] =
            workspace.vars.e[i] + alpha * workspace.delta_vars.e[i];
      }
    } else {
      std::copy_n(workspace.vars.e, s_dim, workspace.next_vars.e);
    }
    add(workspace.miscellaneous_workspace.g_plus_s, workspace.next_vars.e,
        s_dim, workspace.miscellaneous_workspace.g_plus_s_plus_e);
  } else {
    std::copy_n(workspace.miscellaneous_workspace.g_plus_s, s_dim,
                workspace.miscellaneous_workspace.g_plus_s_plus_e);
  }
}

void update_next_dual_vars(const Input &input, const Settings &settings,
                           const double tau, Workspace &workspace,
                           const double merit_delta, const double alpha) {
  const int s_dim = get_s_dim(workspace.model_callback_output->jacobian_g);
  const int y_dim = get_y_dim(workspace.model_callback_output->jacobian_c);

  double allowed_merit_increase =
      std::max(-settings.dual_armijo_factor * merit_delta,
               settings.min_allowed_merit_increase);

  for (int i = 0; i < y_dim; ++i) {
    workspace.next_vars.y[i] =
        workspace.vars.y[i] + alpha * workspace.delta_vars.y[i];
  }

  const double original_y_merit =
      dot(workspace.vars.y, workspace.model_callback_output->c, y_dim);
  const double new_y_merit =
      dot(workspace.next_vars.y, workspace.model_callback_output->c, y_dim);
  const double dm_y = new_y_merit - original_y_merit;

  for (int i = 0; i < s_dim; ++i) {
    workspace.next_vars.z[i] =
        std::max(workspace.vars.z[i] + alpha * workspace.delta_vars.z[i],
                 (1.0 - tau) * workspace.vars.z[i]);
  }

  const double original_z_merit =
      dot(workspace.vars.z, workspace.miscellaneous_workspace.g_plus_s_plus_e,
          s_dim);
  const double new_z_merit =
      dot(workspace.next_vars.z,
          workspace.miscellaneous_workspace.g_plus_s_plus_e, s_dim);
  const double dm_z = new_z_merit - original_z_merit;

  double beta_y = 0.0;
  double beta_z = 0.0;

  if (dm_y <= 0.0) {
    allowed_merit_increase -= dm_y;
    beta_y = 1.0;
  }

  if (dm_z <= 0.0) {
    allowed_merit_increase -= dm_z;
    beta_z = 1.0;
  }

  if (dm_y > 0.0 && dm_z > 0.0) {
    if (dm_y + dm_z < allowed_merit_increase) {
      beta_y = 1.0;
      beta_z = 1.0;
    } else {
      beta_y = 0.5 * allowed_merit_increase / dm_y;
      beta_z = 0.5 * allowed_merit_increase / dm_z;
      if (beta_y > 1.0 && beta_z > 1.0) {
        beta_y = 1.0;
        beta_z = 1.0;
      } else if (beta_y > 1.0) {
        beta_y = 1.0;
        allowed_merit_increase -= dm_y;
        beta_z = std::min(allowed_merit_increase / dm_z, 1.0);
      } else if (beta_z > 0.0) {
        beta_z = 1.0;
        allowed_merit_increase -= dm_z;
        beta_y = std::min(allowed_merit_increase / dm_y, 1.0);
      }
    }
  } else if (dm_y > 0.0 && dm_z <= 0.0) {
    beta_y = std::min(allowed_merit_increase / dm_y, 1.0);
  } else if (dm_z > 0.0 && dm_y <= 0.0) {
    beta_z = std::min(allowed_merit_increase / dm_z, 1.0);
  }

  for (int i = 0; i < y_dim; ++i) {
    workspace.next_vars.y[i] =
        workspace.vars.y[i] +
        beta_y * (workspace.next_vars.y[i] - workspace.vars.y[i]);
  }

  for (int i = 0; i < s_dim; ++i) {
    workspace.next_vars.z[i] =
        workspace.vars.z[i] +
        beta_z * (workspace.next_vars.z[i] - workspace.vars.z[i]);
  }

  if (beta_y > 0.0 || beta_z > 0.0) {
    ModelCallbackInput mci{
        .x = workspace.next_vars.x,
        .y = workspace.next_vars.y,
        .z = workspace.next_vars.z,
        .new_x = false,
        .new_y = true,
        .new_z = true,
    };
    input.model_callback(mci, &workspace.model_callback_output);
  }
}

auto check_derivatives(const Input &input, const Settings &settings,
                       const double tau, Workspace &workspace) -> void {

  const int x_dim =
      workspace.model_callback_output->upper_hessian_lagrangian.rows;
  const int s_dim = get_s_dim(workspace.model_callback_output->jacobian_g);
  const int y_dim = get_y_dim(workspace.model_callback_output->jacobian_c);

  bool has_printed_header = false;

  const auto check_direction = [&](const std::optional<int> var_index) {
    {
      const auto compute_empirical_equality_constraint_slope_errors = [&]() {
        const auto get_perturbed_value =
            [&](const double beta) -> std::vector<double> {
          update_next_primal_vars(input, settings, tau, workspace, beta, true,
                                  false, false);
          std::vector<double> out(y_dim);
          std::copy_n(workspace.model_callback_output->c, y_dim, out.data());
          return out;
        };

        const double h = std::sqrt(std::numeric_limits<double>::epsilon());

        const std::vector<double> mP = get_perturbed_value(h);
        const std::vector<double> mM = get_perturbed_value(-h);

        std::vector<double> theoretical_slopes(y_dim);
        std::fill(theoretical_slopes.begin(), theoretical_slopes.end(), 0.0);
        input.add_Cx_to_y(workspace.model_callback_output->jacobian_c.data,
                          workspace.delta_vars.x, theoretical_slopes.data());

        std::vector<std::tuple<double, double, double, double>> errors(y_dim);
        for (int i = 0; i < y_dim; ++i) {
          const double estimated_slope = (mP[i] - mM[i]) / (2 * h);
          const double absolute_error =
              std::fabs(estimated_slope - theoretical_slopes[i]);
          const double relative_error =
              absolute_error /
              std::max({std::fabs(estimated_slope),
                        std::fabs(theoretical_slopes[i]), 1e-3});
          errors[i] = {relative_error, absolute_error, estimated_slope,
                       theoretical_slopes[i]};
        }
        return errors;
      };
      const auto errors = compute_empirical_equality_constraint_slope_errors();
      for (int i = 0; i < y_dim; ++i) {
        if (std::get<0>(errors[i]) < 0.1) {
          continue;
        }
        if (!has_printed_header) {
          print_derivative_check_log_header();
          has_printed_header = true;
        }
        fmt::print(fg(fmt::color::orange),
                   // clang-format off
                   "{:^10} {:^10} {:^10} {:^10} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g}\n",
                   // clang-format on
                   "", "c", i, var_index.value_or(-1), std::get<0>(errors[i]),
                   std::get<1>(errors[i]), std::get<2>(errors[i]),
                   std::get<3>(errors[i]));
      }
    }

    {
      const auto compute_empirical_inequality_constraint_slope_errors = [&]() {
        const auto get_perturbed_value =
            [&](const double beta) -> std::vector<double> {
          update_next_primal_vars(input, settings, tau, workspace, beta, true,
                                  false, false);
          std::vector<double> out(s_dim);
          std::copy_n(workspace.model_callback_output->g, s_dim, out.data());
          return out;
        };

        const double h = std::sqrt(std::numeric_limits<double>::epsilon());

        const std::vector<double> mP = get_perturbed_value(h);
        const std::vector<double> mM = get_perturbed_value(-h);

        std::vector<double> theoretical_slopes(s_dim);
        std::fill(theoretical_slopes.begin(), theoretical_slopes.end(), 0.0);
        input.add_Gx_to_y(workspace.model_callback_output->jacobian_g.data,
                          workspace.delta_vars.x, theoretical_slopes.data());

        std::vector<std::tuple<double, double, double, double>> errors(s_dim);
        for (int i = 0; i < s_dim; ++i) {
          const double estimated_slope = (mP[i] - mM[i]) / (2 * h);
          const double absolute_error =
              std::fabs(estimated_slope - theoretical_slopes[i]);
          const double relative_error =
              absolute_error /
              std::max({std::fabs(estimated_slope),
                        std::fabs(theoretical_slopes[i]), 1e-3});
          errors[i] = {relative_error, absolute_error, estimated_slope,
                       theoretical_slopes[i]};
        }
        return errors;
      };
      const auto errors =
          compute_empirical_inequality_constraint_slope_errors();
      for (int i = 0; i < s_dim; ++i) {
        if (std::get<0>(errors[i]) < 0.1) {
          continue;
        }
        if (!has_printed_header) {
          print_derivative_check_log_header();
          has_printed_header = true;
        }
        fmt::print(fg(fmt::color::orange),
                   // clang-format off
                   "{:^10} {:^10} {:^10} {:^10} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g}\n",
                   // clang-format on
                   "", "g", i, var_index.value_or(-1), std::get<0>(errors[i]),
                   std::get<1>(errors[i]), std::get<2>(errors[i]),
                   std::get<3>(errors[i]));
      }
    }
    {
      const auto compute_empirical_cost_slope_error = [&]() {
        const auto get_perturbed_value = [&](const double beta) -> double {
          update_next_primal_vars(input, settings, tau, workspace, beta, true,
                                  false, false);
          return workspace.model_callback_output->f;
        };

        const double h = std::sqrt(std::numeric_limits<double>::epsilon());
        const double mP = get_perturbed_value(h);
        const double mM = get_perturbed_value(-h);
        const double estimated_slope = (mP - mM) / (2 * h);
        const double theoretical_slope =
            dot(workspace.model_callback_output->gradient_f,
                workspace.delta_vars.x, x_dim);
        const double absolute_error =
            std::fabs(estimated_slope - theoretical_slope);
        const double relative_error =
            absolute_error / std::max({std::fabs(estimated_slope),
                                       std::fabs(theoretical_slope), 1e-3});
        return std::make_tuple(relative_error, absolute_error, estimated_slope,
                               theoretical_slope);
      };
      const auto error = compute_empirical_cost_slope_error();
      if (std::get<0>(error) > 0.1) {
        if (!has_printed_header) {
          print_derivative_check_log_header();
          has_printed_header = true;
        }
        fmt::print(fg(fmt::color::orange),
                   // clang-format off
                   "{:^10} {:^10} {:^10} {:^10} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g}\n",
                   // clang-format on
                   "", "f", 0, var_index.value_or(-1), std::get<0>(error),
                   std::get<1>(error), std::get<2>(error), std::get<3>(error));
      }
    }
  };

  if (settings.only_check_search_direction_slope) {
    check_direction(std::nullopt);
  } else {
    VariablesWorkspace delta_vars_tmp;
    delta_vars_tmp.reserve(x_dim, s_dim, y_dim);
    std::swap(delta_vars_tmp, workspace.delta_vars);
    std::fill_n(workspace.delta_vars.x, x_dim, 0.0);
    for (int jj = 0; jj < x_dim; ++jj) {
      workspace.delta_vars.x[jj] = 1.0;
      check_direction(jj);
      workspace.delta_vars.x[jj] = 0.0;
    }
    std::swap(delta_vars_tmp, workspace.delta_vars);
  }
  update_next_primal_vars(input, settings, tau, workspace, 0.0, false, false,
                          false);
}

auto get_observed_merit_slope(const Input &input, const Settings &settings,
                              const double eta, const double mu,
                              const double tau, Workspace &workspace)
    -> std::tuple<double, double, double, double> {
  const int s_dim = get_s_dim(workspace.model_callback_output->jacobian_g);
  const int y_dim = get_y_dim(workspace.model_callback_output->jacobian_c);

  const auto compute_empirical_merit_slope =
      [&](const bool update_x, const bool update_s, const bool update_e) {
        const auto get_perturbed_merit = [&](const double beta) -> double {
          update_next_primal_vars(input, settings, tau, workspace, beta,
                                  update_x, update_s, update_e);
          const double ctc =
              squared_norm(workspace.model_callback_output->c, y_dim);
          const double gsetgse = squared_norm(
              workspace.miscellaneous_workspace.g_plus_s_plus_e, s_dim);
          const double sq_constraint_violation_norm = ctc + gsetgse;
          const auto [_mP_f, _mP_s, _mP_c, _mP_g, _mP_e, _mP_aug, mP] =
              merit_function(settings, workspace, workspace.next_vars.s,
                             workspace.vars.y, workspace.vars.z,
                             workspace.next_vars.e, mu, eta,
                             sq_constraint_violation_norm);
          return mP;
        };

        const double h = std::sqrt(std::numeric_limits<double>::epsilon());

        const double mP = get_perturbed_merit(h);
        const double mM = get_perturbed_merit(-h);

        return (mP - mM) / (2 * h);
      };

  const double os_x = compute_empirical_merit_slope(true, false, false);
  const double os_s = compute_empirical_merit_slope(false, true, false);
  const double os_e = compute_empirical_merit_slope(false, false, true);
  const double os = compute_empirical_merit_slope(true, true, true);

  update_next_primal_vars(input, settings, tau, workspace, 0.0, false, false,
                          false);

  return std::make_tuple(os_x, os_s, os_e, os);
}

auto augmented_barrier_lagrangian_slope(
    const Settings &settings, const Workspace &workspace, const double *dx,
    const double *ds, const double *de, const double *dy, const double *dz,
    const double eta, const double sq_constraint_violation_norm)
    -> std::tuple<double, double, double, double> {
  const auto &mco = *workspace.model_callback_output;
  const int x_dim = mco.upper_hessian_lagrangian.rows;
  const int s_dim = get_s_dim(mco.jacobian_g);
  const int y_dim = get_y_dim(mco.jacobian_c);
  const double *gpspe = workspace.miscellaneous_workspace.g_plus_s_plus_e;
  const double abl_slope = dot(workspace.nrhs.x, dx, x_dim) +
                           dot(mco.c, dy, y_dim) + dot(gpspe, dz, s_dim) -
                           eta * sq_constraint_violation_norm;
  const double s_slope =
      dot(workspace.nrhs.s, ds, s_dim) + eta * dot(gpspe, ds, s_dim);
  const double e_slope =
      settings.enable_elastics
          ? dot(workspace.nrhs.e, de, s_dim) + eta * dot(gpspe, de, s_dim)
          : 0.0;
  const double x_slope = abl_slope - s_slope - e_slope;
  return std::make_tuple(x_slope, s_slope, e_slope, abl_slope);
}

auto compute_search_direction(const Input &input, const Settings &settings,
                              const double eta, const double mu,
                              const double tau,
                              const double sq_constraint_violation_norm,
                              Workspace &workspace)
    -> std::tuple<const double *, const double *, const double *,
                  const double *, const double *, double, double, double,
                  double> {
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
  double *LT_data = workspace.csd_workspace.LT_data;
  double *D_diag = workspace.csd_workspace.D_diag;
  double *b = workspace.csd_workspace.rhs_block_3x3;
  double *residual = workspace.csd_workspace.residual;
  double *v = workspace.csd_workspace.sol_block_3x3;
  double *u = workspace.csd_workspace.iterative_refinement_error_sol;

  double *bx = b;
  double *by = bx + x_dim;
  double *bz = by + y_dim;

  double *vx = v;
  double *vy = vx + x_dim;
  double *vz = vy + y_dim;

  for (int i = 0; i < s_dim; ++i) {
    w[i] = s[i] / z[i];
  }

  constexpr double r1 = 0.0;
  const double eta_inv = 1.0 / eta;
  const double r3p = settings.enable_elastics ? eta_inv + 1.0 / rho : eta_inv;

  input.ldlt_factor(H_data, C_data, G_data, w, r1, eta_inv, r3p, LT_data,
                    D_diag);

  std::copy_n(grad_f, x_dim, rx);

  input.add_CTx_to_y(C_data, y, rx);
  input.add_GTx_to_y(G_data, z, rx);

  std::copy_n(c, y_dim, ry);
  std::copy_n(gpspe, s_dim, rz);

  std::copy_n(rx, x_dim, bx);
  std::copy_n(c, y_dim, by);

  if (settings.enable_elastics) {
    for (int i = 0; i < s_dim; ++i) {
      rs[i] = z[i] - mu / s[i];
      re[i] = rho * e[i] + z[i];
      bz[i] = rz[i] - re[i] / rho - w[i] * rs[i];
    }
  } else {
    for (int i = 0; i < s_dim; ++i) {
      rs[i] = z[i] - mu / s[i];
      bz[i] = rz[i] - w[i] * rs[i];
    }
  }

  for (int i = 0; i < dim_3x3; ++i) {
    b[i] = -b[i];
  }

  input.ldlt_solve(LT_data, D_diag, b, v);

  for (int j = 0; j < settings.num_iterative_refinement_steps; ++j) {
    // We do an iterative refinement step:
    // res = Kv - b
    // Ku = res => Ku = Kv - b => K(v - u) = b

    double *res_x = residual;
    double *res_y = res_x + x_dim;
    double *res_z = res_y + y_dim;

    for (int i = 0; i < dim_3x3; ++i) {
      residual[i] = -b[i];
    }

    input.add_Kx_to_y(H_data, C_data, G_data, w, r1, eta_inv, r3p, vx, vy, vz,
                      res_x, res_y, res_z);

    input.ldlt_solve(LT_data, D_diag, residual, u);

    for (int i = 0; i < dim_3x3; ++i) {
      v[i] -= u[i];
    }
  }

  {
    auto ptr = v;
    std::copy_n(ptr, x_dim, dx);
    ptr += x_dim;
    std::copy_n(ptr, y_dim, dy);
    ptr += y_dim;
    std::copy_n(ptr, s_dim, dz);
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

  const auto [ms_x, ms_s, ms_e, ms] =
      augmented_barrier_lagrangian_slope(settings, workspace, dx, ds, de, dy,
                                         dz, eta, sq_constraint_violation_norm);

  kkt_error = 0.0;

  for (int i = 0; i < x_dim; ++i) {
    kkt_error = std::max(kkt_error, std::fabs(rx[i]));
  }
  for (int i = 0; i < y_dim; ++i) {
    kkt_error = std::max(kkt_error, std::fabs(c[i]));
  }
  for (int i = 0; i < s_dim; ++i) {
    kkt_error = std::max(
        kkt_error,
        std::fabs(workspace.miscellaneous_workspace.g_plus_s_plus_e[i]));
  }

  const auto alpha_s_max =
      get_alpha_s_max(s_dim, tau, workspace.vars.s, workspace.delta_vars.s);

  if (settings.print_search_direction_logs) {
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

    print_search_direction_log_header();

    double *tmp_x = workspace.next_vars.x;
    double *tmp_y = workspace.next_vars.y;
    double *tmp_s = workspace.next_vars.s;
    std::fill_n(tmp_x, x_dim, 0.0);
    input.add_upper_symmetric_Hx_to_y(H_data, dx, tmp_x);
    const double dxTHdx = dot(tmp_x, dx, x_dim);
    std::fill_n(tmp_y, y_dim, 0.0);
    input.add_Cx_to_y(C_data, dx, tmp_y);
    const double Cdx2 = squared_norm(tmp_y, y_dim);
    double Winvds = 0.0;
    for (int i = 0; i < s_dim; ++i) {
      Winvds += ds[i] * ds[i] / w[i];
    }
    if (settings.enable_elastics) {
      for (int i = 0; i < s_dim; ++i) {
        tmp_s[i] = ds[i] + de[i];
      }
    } else {
      std::copy_n(ds, s_dim, tmp_s);
    }
    input.add_Gx_to_y(G_data, dx, tmp_s);
    const double Gdepdspde2 = squared_norm(tmp_s, s_dim);
    double ms_v2 = -dxTHdx - Winvds - eta * (Cdx2 + Gdepdspde2);
    if (settings.enable_elastics) {
      ms_v2 -= settings.elastic_var_cost_coeff * squared_norm(de, s_dim);
    }

    const auto [os_x, os_s, os_e, os] =
        get_observed_merit_slope(input, settings, eta, mu, tau, workspace);

    const double re_norm = settings.enable_elastics ? norm(re, s_dim) : -1.0;
    fmt::print(fg(fmt::color::green),
               // clang-format off
                   "{:^10} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g}\n",
               // clang-format on
               "", lin_sys_error, alpha_s_max, ms, ms_v2, os, ms_x, os_x, ms_s,
               os_s, ms_e, os_e, norm(rx, x_dim), norm(rs, s_dim), re_norm);
    const bool suspicious_derivatives =
        (os_x > 0.0 && ms_x < 0.0) || (os > 0.0 && ms < 0.0) ||
        (std::fabs(ms_x - os_x) /
             std::max({std::fabs(ms_x), std::fabs(os_x), 1e-12}) >
         0.5);
    if (settings.print_derivative_check_logs && suspicious_derivatives)
      check_derivatives(input, settings, tau, workspace);
  }

  return std::make_tuple(dx, ds, dy, dz, de, ms, alpha_s_max, kkt_error,
                         lin_sys_error);
}

auto do_line_search(const Input &input, const Settings &settings,
                    const double eta, const double mu, const double tau,
                    const double sq_constraint_violation_norm,
                    const double merit_slope, const double alpha_s_max,
                    int &total_ls_iterations, Workspace &workspace)
    -> std::tuple<double, double, double, double> {
  const int s_dim = get_s_dim(workspace.model_callback_output->jacobian_g);
  const int y_dim = get_y_dim(workspace.model_callback_output->jacobian_c);

  const auto [m0_f, m0_s, m0_c, m0_g, m0_e, m0_aug, m0] = merit_function(
      settings, workspace, workspace.vars.s, workspace.vars.y, workspace.vars.z,
      workspace.vars.e, mu, eta, sq_constraint_violation_norm);

  bool ls_succeeded = false;
  double alpha = settings.start_ls_with_alpha_s_max ? alpha_s_max : 1.0;
  double merit_delta = std::numeric_limits<double>::signaling_NaN();
  double constraint_violation_ratio =
      std::numeric_limits<double>::signaling_NaN();

  if (settings.print_line_search_logs) {
    print_line_search_log_header();
  }

  do {
    update_next_primal_vars(input, settings, tau, workspace, alpha, true, true,
                            true);
    const double next_ctc =
        squared_norm(workspace.model_callback_output->c, y_dim);
    const double next_gsetgse =
        squared_norm(workspace.miscellaneous_workspace.g_plus_s_plus_e, s_dim);
    const double next_sq_constraint_violation_norm = next_ctc + next_gsetgse;

    // NOTE: the line search is only over the primal variables, so we cannot
    // use next_vars.y and next_vars.z.
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

    merit_delta = dm_f + dm_s + dm_c + dm_g + dm_e + dm_aug;

    constraint_violation_ratio = next_sq_constraint_violation_norm /
                                 std::max(sq_constraint_violation_norm, 1e-12);

    if (settings.print_line_search_logs) {
      fmt::print(fg(fmt::color::yellow),
                 // clang-format off
                       "{:^10} {:^+10} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g}\n",
                 // clang-format on
                 "", total_ls_iterations, alpha, m, m_f, std::sqrt(next_ctc),
                 std::sqrt(next_gsetgse), merit_delta, merit_delta / alpha,
                 dm_f, dm_s, dm_c, dm_g, dm_e, dm_aug);
    }

    ++total_ls_iterations;

    if (merit_slope > settings.min_merit_slope_to_skip_line_search ||
        merit_delta < settings.armijo_factor * merit_slope * alpha) {
      ls_succeeded = true;
      break;
    }

    alpha *= settings.line_search_factor;
  } while (alpha > settings.line_search_min_step_size &&
           total_ls_iterations < settings.max_ls_iterations);

  if (alpha <= settings.line_search_min_step_size) {
    alpha /= settings.line_search_factor;
  }

  update_next_dual_vars(input, settings, tau, workspace, merit_delta, alpha);

  return std::make_tuple(ls_succeeded, alpha, m0, constraint_violation_ratio);
}

auto check_settings(const Settings &settings) -> bool {
  if (settings.enable_elastics && settings.elastic_var_cost_coeff <= 0.0) {
    return false;
  }
  if (settings.max_penalty_parameter < settings.initial_penalty_parameter) {
    return false;
  }
  if (settings.print_line_search_logs && !settings.print_logs) {
    return false;
  }
  if (settings.print_search_direction_logs && !settings.print_logs) {
    return false;
  }
  if (settings.print_derivative_check_logs && !settings.print_logs) {
    return false;
  }
  return true;
}

} // namespace

auto solve(const Input &input, const Settings &settings, Workspace &workspace)
    -> Output {
  {
    ModelCallbackInput mci{
        .x = workspace.vars.x,
        .y = workspace.vars.y,
        .z = workspace.vars.z,
        .new_x = true,
        .new_y = true,
        .new_z = true,
    };

    input.model_callback(mci, &workspace.model_callback_output);
  }

  if (!check_settings(settings)) {
    if (settings.assert_checks_pass) {
      assert(false && "check_settings returned false.");
    } else {
      return Output{
          .exit_status = Status::FAILED_CHECK,
          .num_iterations = 0,
          .max_primal_violation = std::numeric_limits<double>::signaling_NaN(),
          .max_dual_violation = -std::numeric_limits<double>::signaling_NaN(),
      };
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
        return Output{
            .exit_status = Status::FAILED_CHECK,
            .num_iterations = 0,
            .max_primal_violation =
                std::numeric_limits<double>::signaling_NaN(),
            .max_dual_violation = -std::numeric_limits<double>::signaling_NaN(),
        };
      }
    }
    if (workspace.vars.z[i] <= 0.0) {
      if (settings.assert_checks_pass) {
        assert(false && "workspace.vars.z[i] <= 0.0.");
      } else {
        return Output{
            .exit_status = Status::FAILED_CHECK,
            .num_iterations = 0,
            .max_primal_violation =
                std::numeric_limits<double>::signaling_NaN(),
            .max_dual_violation = -std::numeric_limits<double>::signaling_NaN(),
        };
      }
    }
  }

  add(workspace.model_callback_output->g, workspace.vars.s, s_dim,
      workspace.miscellaneous_workspace.g_plus_s);

  if (settings.enable_elastics) {
    add(workspace.miscellaneous_workspace.g_plus_s, workspace.vars.e, s_dim,
        workspace.miscellaneous_workspace.g_plus_s_plus_e);
  } else {
    std::copy_n(workspace.miscellaneous_workspace.g_plus_s, s_dim,
                workspace.miscellaneous_workspace.g_plus_s_plus_e);
  }

  double eta = settings.initial_penalty_parameter;
  double mu = settings.initial_mu;
  const double tau = settings.tau;

  int total_ls_iterations = 0;

  for (int iteration = 0; iteration < settings.max_iterations; ++iteration) {
    const double f0 = workspace.model_callback_output->f;

    const double ctc = squared_norm(workspace.model_callback_output->c, y_dim);
    const double gsetgse =
        squared_norm(workspace.miscellaneous_workspace.g_plus_s_plus_e, s_dim);

    const double sq_constraint_violation_norm = ctc + gsetgse;

    const auto [dx, ds, dy, dz, de, merit_slope, alpha_s_max, kkt_error,
                lin_sys_error] =
        compute_search_direction(input, settings, eta, mu, tau,
                                 sq_constraint_violation_norm, workspace);

    const bool succeeded =
        iteration >= settings.min_iterations_for_convergence &&
        (kkt_error < settings.max_kkt_violation ||
         std::fabs(merit_slope) < settings.max_merit_slope);

    const bool hit_ls_iteration_limit =
        total_ls_iterations >= settings.max_ls_iterations;

    if (settings.print_logs && (succeeded || hit_ls_iteration_limit)) {
      if (iteration == 0 || settings.print_line_search_logs ||
          settings.print_search_direction_logs) {
        print_log_header();
      }
      const double de_norm = settings.enable_elastics ? norm(de, s_dim) : -1.0;
      fmt::print(fg(fmt::color::red),
                 // clang-format off
                       "{:^+10} {:^10} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^10} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g}\n",
                 // clang-format on
                 iteration, "", workspace.model_callback_output->f,
                 std::sqrt(ctc), std::sqrt(gsetgse), "", norm(dx, x_dim),
                 norm(ds, s_dim), norm(dy, y_dim), norm(dz, s_dim), de_norm, mu,
                 eta, tau, kkt_error);
    }

    if (succeeded) {
      return Output{
          .exit_status = Status::SOLVED,
          .num_iterations = iteration,
          .max_primal_violation =
              inf_norm(workspace.model_callback_output->c, y_dim),
          .max_dual_violation = inf_norm(workspace.nrhs.x, x_dim),
      };
    }

    if (hit_ls_iteration_limit) {
      return Output{
          .exit_status = Status::LINE_SEARCH_ITERATION_LIMIT,
          .num_iterations = iteration,
          .max_primal_violation =
              inf_norm(workspace.model_callback_output->c, y_dim),
          .max_dual_violation = inf_norm(workspace.nrhs.x, x_dim),
      };
    }

    const auto [ls_succeeded, alpha, m0, constraint_violation_ratio] =
        do_line_search(input, settings, eta, mu, tau,
                       sq_constraint_violation_norm, merit_slope, alpha_s_max,
                       total_ls_iterations, workspace);

    if (settings.print_logs) {
      if (iteration == 0 || settings.print_line_search_logs) {
        print_log_header();
      }

      const double de_norm = settings.enable_elastics ? norm(de, s_dim) : -1.0;
      fmt::print(fg(fmt::color::red),
                 // clang-format off
                       "{:^+10} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g}  {:^+10.4g} {:^+10.4g}\n",
                 // clang-format on
                 iteration, alpha, f0, std::sqrt(ctc), std::sqrt(gsetgse), m0,
                 norm(dx, x_dim), norm(ds, s_dim), norm(dy, y_dim),
                 norm(dz, s_dim), de_norm, mu, eta, tau, kkt_error);
    }

    if (settings.enable_line_search_failures && !ls_succeeded) {
      return Output{
          .exit_status = Status::LINE_SEARCH_FAILURE,
          .num_iterations = iteration,
          .max_primal_violation =
              inf_norm(workspace.model_callback_output->c, y_dim),
          .max_dual_violation = inf_norm(workspace.nrhs.x, x_dim),
      };
    }

    std::swap(workspace.vars, workspace.next_vars);

    if (input.timeout_callback()) {
      return Output{
          .exit_status = Status::TIMEOUT,
          .num_iterations = iteration,
          .max_primal_violation =
              inf_norm(workspace.model_callback_output->c, y_dim),
          .max_dual_violation = inf_norm(workspace.nrhs.x, x_dim),
      };
    }

    mu = std::max(mu * settings.mu_update_factor, settings.mu_min);

    if (constraint_violation_ratio >
        settings.min_acceptable_constraint_violation_ratio) {
      eta = std::min(eta * settings.penalty_parameter_increase_factor,
                     settings.max_penalty_parameter);
    } else {
      eta = std::min(eta * settings.penalty_parameter_decrease_factor,
                     settings.max_penalty_parameter);
    }
  }

  return Output{
      .exit_status = Status::ITERATION_LIMIT,
      .num_iterations = settings.max_iterations,
      .max_primal_violation =
          inf_norm(workspace.model_callback_output->c, y_dim),
      .max_dual_violation = inf_norm(workspace.nrhs.x, x_dim),
  };
}

} // namespace sip
