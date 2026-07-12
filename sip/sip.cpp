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

auto mean_penalty_parameter(const Workspace &workspace, const int s_dim,
                            const int y_dim) -> double {
  double sum = 0.0;
  for (int i = 0; i < y_dim; ++i) {
    sum += workspace.penalties.y[i];
  }
  for (int i = 0; i < s_dim; ++i) {
    sum += workspace.penalties.z[i];
  }
  const int dim = s_dim + y_dim;
  return dim == 0 ? 0.0 : sum / dim;
}

auto max_primal_violation(const Input &input) -> double {
  const int s_dim = input.dimensions.s_dim;
  const int y_dim = input.dimensions.y_dim;
  return std::max(max_abs_or_inf(input.get_c(), y_dim),
                  max_positive_or_inf(input.get_g(), s_dim));
}

auto merit_function(const Input &input, const Workspace &workspace,
                    const double *s, const double *y, const double *z,
                    const double mu)
    -> std::tuple<double, double, double, double, double, double> {
  const int s_dim = input.dimensions.s_dim;
  const int y_dim = input.dimensions.y_dim;
  const double *c = input.get_c();
  const double *gps = workspace.miscellaneous_workspace.g_plus_s;
  const double s_term = -mu * sum_of_logs(s, s_dim);
  const double c_term = dot(c, y, y_dim);
  const double g_term = dot(gps, z, s_dim);
  const double barrier_lagrangian = input.get_f() + s_term + c_term + g_term;
  const double aug_term =
      0.5 * (weighted_squared_norm(c, workspace.penalties.y, y_dim) +
             weighted_squared_norm(gps, workspace.penalties.z, s_dim));
  const double merit = barrier_lagrangian + aug_term;
  return std::make_tuple(input.get_f(), s_term, c_term, g_term, aug_term,
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
             "{:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10}\n",
             // clang-format on
             "iteration", "alpha", "f", "|c|", "|g+s|", "merit", "|dx|", "|ds|",
             "|dy|", "|dz|", "mu", "eta", "tau", "psi", "n_reg", "max_sz",
             "dual_res", "kkt_error");
}

void print_search_direction_log_header() {
  fmt::print(fmt::emphasis::bold | fg(fmt::color::green),
             // clang-format off
             "{:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10}\n",
             // clang-format on
             "", "linsys_res", "alpha_s_m", "m_slope", "m_slope_v2",
             "obs_slope", "m_sl_x", "obs_sl_x", "m_sl_s", "obs_sl_s",
             "|nrhs_x|", "|nrhs_s|");
}

void print_line_search_log_header() {
  fmt::print(fmt::emphasis::bold | fg(fmt::color::yellow),
             // clang-format off
             "{:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10}\n",
             // clang-format on
             "", "ls_iteration", "alpha", "merit", "f", "|c|", "|g+s|", "dm",
             "dm/alpha", "dm[f]", "dm[s]", "dm[c]", "dm[g]", "dm[aug]");
}

void print_derivative_check_log_header() {
  fmt::print(fmt::emphasis::bold | fg(fmt::color::orange),
             // clang-format off
             "{:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10}\n",
             // clang-format on
             "", "f/c/g", "out_index", "var_index", "rel_error", "abs_error",
             "est_slope", "theo_slope");
}

auto increased_regularization(const Settings &settings, const double psi)
    -> double {
  const auto &regularization = settings.regularization;
  if (psi < regularization.first_positive) {
    return regularization.first_positive;
  }
  return std::min(psi * regularization.increase_factor, regularization.maximum);
}

auto decreased_regularization(const Settings &settings, const double psi)
    -> double {
  const auto &regularization = settings.regularization;
  const double decreased = psi * regularization.decrease_factor;
  return decreased < regularization.first_positive ? 0.0 : decreased;
}

struct TerminationChecks {
  bool solved;
  bool stalled;
  bool advance_barrier;
};

auto cost_change_satisfied(const Settings &settings,
                           const std::optional<double> previous_cost,
                           const double current_cost) -> bool {
  if (!settings.termination.enable_cost_change_termination ||
      !previous_cost.has_value() || !std::isfinite(*previous_cost) ||
      !std::isfinite(current_cost)) {
    return false;
  }

  const double cost_change = std::fabs(current_cost - *previous_cost);
  const double cost_scale = std::max(1.0, std::fabs(*previous_cost));
  const double max_cost_change =
      settings.termination.max_cost_change +
      settings.termination.max_relative_cost_change * cost_scale;
  return cost_change <= max_cost_change;
}

auto check_termination(const Settings &settings,
                       const std::optional<double> previous_cost,
                       const double current_cost, const double merit_slope,
                       const double dual_residual,
                       const double max_constraint_violation,
                       const double max_complementarity,
                       const double duality_gap) -> TerminationChecks {
  const bool duality_gap_satisfied =
      duality_gap <= settings.termination.max_duality_gap;
  const bool primal_feasibility_satisfied =
      max_constraint_violation < settings.termination.max_constraint_violation;
  const bool dual_residual_satisfied =
      dual_residual < settings.termination.max_dual_residual;
  const bool complementarity_satisfied =
      max_complementarity < settings.termination.max_complementarity_gap;
  const bool merit_slope_too_small =
      merit_slope > -settings.termination.max_merit_slope;
  const bool cost_change_ok =
      cost_change_satisfied(settings, previous_cost, current_cost);

  const bool kkt_optimality_satisfied = dual_residual_satisfied &&
                                        primal_feasibility_satisfied &&
                                        complementarity_satisfied;
  const bool cost_change_optimality_satisfied =
      cost_change_ok && primal_feasibility_satisfied;
  const bool advance_barrier = merit_slope_too_small &&
                               primal_feasibility_satisfied &&
                               dual_residual_satisfied &&
                               !complementarity_satisfied;

  return TerminationChecks{
      // TODO(joao): probably only consider the duality gap in the first case...
      .solved = duality_gap_satisfied &&
                (kkt_optimality_satisfied || cost_change_optimality_satisfied),
      .stalled = merit_slope_too_small && !advance_barrier,
      .advance_barrier = advance_barrier,
  };
}

auto update_penalty_parameters(const Input &input, const Settings &settings,
                               Workspace &workspace) -> bool {
  const int s_dim = input.dimensions.s_dim;
  const int y_dim = input.dimensions.y_dim;
  bool any_increased = false;

  const double *old_c = workspace.nrhs.y;
  const double *old_gps = workspace.nrhs.z;
  const double *new_c = input.get_c();
  const double *new_gps = workspace.miscellaneous_workspace.g_plus_s;
  const double min_ratio =
      settings.penalty.min_acceptable_constraint_violation_ratio;

  for (int i = 0; i < y_dim; ++i) {
    const double improvement_ratio =
        new_c[i] * new_c[i] / std::max(old_c[i] * old_c[i], 1e-12);
    if (improvement_ratio > min_ratio) {
      workspace.penalties.y[i] =
          std::min(workspace.penalties.y[i] *
                       settings.penalty.penalty_parameter_increase_factor,
                   settings.penalty.max_penalty_parameter);
      any_increased = true;
    } else {
      workspace.penalties.y[i] =
          std::min(workspace.penalties.y[i] *
                       settings.penalty.penalty_parameter_decrease_factor,
                   settings.penalty.max_penalty_parameter);
    }
  }

  for (int i = 0; i < s_dim; ++i) {
    const double improvement_ratio =
        new_gps[i] * new_gps[i] / std::max(old_gps[i] * old_gps[i], 1e-12);
    if (improvement_ratio > min_ratio) {
      workspace.penalties.z[i] =
          std::min(workspace.penalties.z[i] *
                       settings.penalty.penalty_parameter_increase_factor,
                   settings.penalty.max_penalty_parameter);
      any_increased = true;
    } else {
      workspace.penalties.z[i] =
          std::min(workspace.penalties.z[i] *
                       settings.penalty.penalty_parameter_decrease_factor,
                   settings.penalty.max_penalty_parameter);
    }
  }

  return any_increased;
}

void update_next_primal_vars(const Input &input, const double tau,
                             Workspace &workspace, const double alpha,
                             const bool update_x, const bool update_s) {
  const int x_dim = input.dimensions.x_dim;
  const int s_dim = input.dimensions.s_dim;

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
  input.model_callback(mci);

  if (update_s) {
    for (int i = 0; i < s_dim; ++i) {
      workspace.next_vars.s[i] =
          std::max(workspace.vars.s[i] + alpha * workspace.delta_vars.s[i],
                   (1.0 - tau) * workspace.vars.s[i]);
    }
  } else {
    std::copy_n(workspace.vars.s, s_dim, workspace.next_vars.s);
  }

  add(input.get_g(), workspace.next_vars.s, s_dim,
      workspace.miscellaneous_workspace.g_plus_s);
}

void update_next_dual_vars(const Input &input, const double tau,
                           Workspace &workspace, const double alpha) {
  const int s_dim = input.dimensions.s_dim;
  const int y_dim = input.dimensions.y_dim;

  for (int i = 0; i < y_dim; ++i) {
    workspace.next_vars.y[i] =
        workspace.vars.y[i] + alpha * workspace.delta_vars.y[i];
  }

  for (int i = 0; i < s_dim; ++i) {
    workspace.next_vars.z[i] =
        std::max(workspace.vars.z[i] + alpha * workspace.delta_vars.z[i],
                 (1.0 - tau) * workspace.vars.z[i]);
  }

  ModelCallbackInput mci{
      .x = workspace.next_vars.x,
      .y = workspace.next_vars.y,
      .z = workspace.next_vars.z,
      .new_x = false,
      .new_y = true,
      .new_z = true,
  };
  input.model_callback(mci);
}

auto check_derivatives(const Input &input, const Settings &settings,
                       const double tau, Workspace &workspace) -> void {

  const int x_dim = input.dimensions.x_dim;
  const int s_dim = input.dimensions.s_dim;
  const int y_dim = input.dimensions.y_dim;

  bool has_printed_header = false;

  const auto check_direction = [&](const std::optional<int> var_index) {
    {
      const auto compute_empirical_equality_constraint_slope_errors = [&]() {
        const auto get_perturbed_value =
            [&](const double beta) -> std::vector<double> {
          update_next_primal_vars(input, tau, workspace, beta, true, false);
          std::vector<double> out(y_dim);
          std::copy_n(input.get_c(), y_dim, out.data());
          return out;
        };

        const double h = std::sqrt(std::numeric_limits<double>::epsilon());

        const std::vector<double> mP = get_perturbed_value(h);
        const std::vector<double> mM = get_perturbed_value(-h);

        std::vector<double> theoretical_slopes(y_dim);
        std::fill(theoretical_slopes.begin(), theoretical_slopes.end(), 0.0);
        input.add_Cx_to_y(workspace.delta_vars.x, theoretical_slopes.data());

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
          update_next_primal_vars(input, tau, workspace, beta, true, false);
          std::vector<double> out(s_dim);
          std::copy_n(input.get_g(), s_dim, out.data());
          return out;
        };

        const double h = std::sqrt(std::numeric_limits<double>::epsilon());

        const std::vector<double> mP = get_perturbed_value(h);
        const std::vector<double> mM = get_perturbed_value(-h);

        std::vector<double> theoretical_slopes(s_dim);
        std::fill(theoretical_slopes.begin(), theoretical_slopes.end(), 0.0);
        input.add_Gx_to_y(workspace.delta_vars.x, theoretical_slopes.data());

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
          update_next_primal_vars(input, tau, workspace, beta, true, false);
          return input.get_f();
        };

        const double h = std::sqrt(std::numeric_limits<double>::epsilon());
        const double mP = get_perturbed_value(h);
        const double mM = get_perturbed_value(-h);
        const double estimated_slope = (mP - mM) / (2 * h);
        const double theoretical_slope =
            dot(input.get_grad_f(), workspace.delta_vars.x, x_dim);
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

  if (settings.logging.only_check_search_direction_slope) {
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
  update_next_primal_vars(input, tau, workspace, 0.0, false, false);
}

auto get_observed_merit_slope(const Input &input, const double mu,
                              const double tau, Workspace &workspace)
    -> std::tuple<double, double, double> {
  const auto compute_empirical_merit_slope = [&](const bool update_x,
                                                 const bool update_s) {
    const auto get_perturbed_merit = [&](const double beta) -> double {
      update_next_primal_vars(input, tau, workspace, beta, update_x, update_s);
      const auto [_mP_f, _mP_s, _mP_c, _mP_g, _mP_aug, mP] =
          merit_function(input, workspace, workspace.next_vars.s,
                         workspace.vars.y, workspace.vars.z, mu);
      return mP;
    };

    const double h = std::sqrt(std::numeric_limits<double>::epsilon());

    const double mP = get_perturbed_merit(h);
    const double mM = get_perturbed_merit(-h);

    return (mP - mM) / (2 * h);
  };

  const double os_x = compute_empirical_merit_slope(true, false);
  const double os_s = compute_empirical_merit_slope(false, true);
  const double os = compute_empirical_merit_slope(true, true);

  update_next_primal_vars(input, tau, workspace, 0.0, false, false);

  return std::make_tuple(os_x, os_s, os);
}

struct FixedDualMeritSlope {
  double x;
  double s;
  double total;
};

auto fixed_dual_merit_slope(const Input &input, const Workspace &workspace,
                            const double *dx, const double *ds)
    -> FixedDualMeritSlope {
  const int x_dim = input.dimensions.x_dim;
  const int s_dim = input.dimensions.s_dim;
  const int y_dim = input.dimensions.y_dim;
  const double *c = input.get_c();
  const double *gps = workspace.miscellaneous_workspace.g_plus_s;

  double *tmp_y = workspace.next_vars.y;
  double *tmp_s = workspace.next_vars.s;

  std::fill_n(tmp_y, y_dim, 0.0);
  input.add_Cx_to_y(dx, tmp_y);

  std::fill_n(tmp_s, s_dim, 0.0);
  input.add_Gx_to_y(dx, tmp_s);

  const double x_slope =
      dot(workspace.nrhs.x, dx, x_dim) +
      weighted_dot(c, workspace.penalties.y, tmp_y, y_dim) +
      weighted_dot(gps, workspace.penalties.z, tmp_s, s_dim);

  const double s_slope =
      dot(workspace.nrhs.s, ds, s_dim) +
      weighted_dot(gps, workspace.penalties.z, ds, s_dim);

  return {.x = x_slope, .s = s_slope, .total = x_slope + s_slope};
}

auto quadratic_model_merit_slope(const Input &input, const Workspace &workspace,
                                 const double *dx, const double *ds,
                                 const double psi) -> double {
  const int x_dim = input.dimensions.x_dim;
  const int s_dim = input.dimensions.s_dim;
  const int y_dim = input.dimensions.y_dim;

  double *tmp_x = workspace.next_vars.x;
  double *tmp_y = workspace.next_vars.y;
  double *tmp_s = workspace.next_vars.s;

  std::fill_n(tmp_x, x_dim, 0.0);
  input.add_Hx_to_y(dx, tmp_x);
  for (int i = 0; i < x_dim; ++i) {
    tmp_x[i] += psi * dx[i];
  }
  const double dxTHdx = dot(tmp_x, dx, x_dim);

  std::fill_n(tmp_y, y_dim, 0.0);
  input.add_Cx_to_y(dx, tmp_y);

  std::copy_n(ds, s_dim, tmp_s);
  input.add_Gx_to_y(dx, tmp_s);

  double Winvds = 0.0;
  for (int i = 0; i < s_dim; ++i) {
    Winvds += ds[i] * ds[i] / workspace.csd_workspace.w[i];
  }

  return -dxTHdx - Winvds -
         weighted_squared_norm(tmp_y, workspace.penalties.y, y_dim) -
         weighted_squared_norm(tmp_s, workspace.penalties.z, s_dim);
}

auto compute_search_direction(const Input &input, const Settings &settings,
                              const double mu, double &psi, const double tau,
                              Workspace &workspace)
    -> std::tuple<bool, const double *, const double *, const double *,
                  const double *, double, double, double, double, double,
                  double, double, double, int> {
  const int x_dim = input.dimensions.x_dim;
  const int s_dim = input.dimensions.s_dim;
  const int y_dim = input.dimensions.y_dim;
  const int dim_3x3 = x_dim + s_dim + y_dim;

  double dual_residual = 0.0;
  double max_constraint_violation = 0.0;
  double max_complementarity = 0.0;
  double kkt_error = 0.0;
  double duality_gap = 0.0;
  double lin_sys_error = std::numeric_limits<double>::signaling_NaN();

  const double *s = workspace.vars.s;
  const double *y = workspace.vars.y;
  const double *z = workspace.vars.z;

  const double *grad_f = input.get_grad_f();

  const double *c = input.get_c();
  const double *gps = workspace.miscellaneous_workspace.g_plus_s;

  double *dx = workspace.delta_vars.x;
  double *ds = workspace.delta_vars.s;
  double *dy = workspace.delta_vars.y;
  double *dz = workspace.delta_vars.z;

  double *rx = workspace.nrhs.x;
  double *rs = workspace.nrhs.s;
  double *ry = workspace.nrhs.y;
  double *rz = workspace.nrhs.z;

  double *w = workspace.csd_workspace.w;
  double *r2 = workspace.csd_workspace.r2;
  double *r3 = workspace.csd_workspace.r3;
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
    w[i] = std::clamp(s[i] / z[i], 1e-18, 1e18);
  }

  for (int i = 0; i < y_dim; ++i) {
    r2[i] = 1.0 / workspace.penalties.y[i];
  }
  for (int i = 0; i < s_dim; ++i) {
    r3[i] = 1.0 / workspace.penalties.z[i];
  }

  int num_regularization_increases = 0;
  bool factorization_ok = false;
  for (int attempt = 0; attempt < settings.regularization.max_attempts;
       ++attempt) {
    factorization_ok = input.factor(w, psi, r2, r3);
    if (factorization_ok) {
      break;
    }
    const double next_psi = increased_regularization(settings, psi);
    if (next_psi <= psi || next_psi > settings.regularization.maximum) {
      break;
    }
    psi = next_psi;
    ++num_regularization_increases;
  }

  if (!factorization_ok) {
    return std::make_tuple(false, dx, ds, dy, dz, 0.0, 0.0,
                           std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::infinity(), 0.0,
                           lin_sys_error, num_regularization_increases);
  }

  std::copy_n(grad_f, x_dim, rx);

  input.add_CTx_to_y(y, rx);
  input.add_GTx_to_y(z, rx);

  if (std::isfinite(settings.termination.max_duality_gap)) {
    const double *g = input.get_g();
    duality_gap =
        dot(workspace.vars.x, rx, x_dim) - dot(c, y, y_dim) - dot(g, z, s_dim);
    duality_gap = std::fabs(duality_gap);
    if (std::isnan(duality_gap)) {
      duality_gap = std::numeric_limits<double>::infinity();
    }
  }

  std::copy_n(c, y_dim, ry);
  std::copy_n(gps, s_dim, rz);

  std::copy_n(rx, x_dim, bx);
  std::copy_n(c, y_dim, by);

  for (int i = 0; i < s_dim; ++i) {
    rs[i] = z[i] - mu / s[i];
    bz[i] = rz[i] - w[i] * rs[i];
  }

  for (int i = 0; i < dim_3x3; ++i) {
    b[i] = -b[i];
  }

  input.solve(b, v);

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

    input.add_Kx_to_y(w, psi, r2, r3, vx, vy, vz, res_x, res_y, res_z);

    input.solve(residual, u);

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

  const auto fixed_dual_ms = fixed_dual_merit_slope(input, workspace, dx, ds);

  for (int i = 0; i < y_dim; ++i) {
    max_constraint_violation =
        std::max(max_constraint_violation, std::fabs(c[i]));
  }

  for (int i = 0; i < s_dim; ++i) {
    const double abs_gps =
        std::fabs(workspace.miscellaneous_workspace.g_plus_s[i]);
    max_constraint_violation =
        std::isnan(abs_gps) ? std::numeric_limits<double>::infinity()
                            : std::max(max_constraint_violation, abs_gps);
    const double sz = s[i] * z[i];
    max_complementarity = std::isnan(sz)
                              ? std::numeric_limits<double>::infinity()
                              : std::max(max_complementarity, sz);
  }

  for (int i = 0; i < x_dim; ++i) {
    const double abs_rx = std::fabs(rx[i]);
    dual_residual = std::isnan(abs_rx) ? std::numeric_limits<double>::infinity()
                                       : std::max(dual_residual, abs_rx);
  }

  kkt_error = dual_residual;
  kkt_error = std::max(kkt_error, max_constraint_violation);
  kkt_error = std::max(kkt_error, max_complementarity);
  if (std::isnan(kkt_error)) {
    kkt_error = std::numeric_limits<double>::infinity();
  }

  const auto alpha_s_max =
      get_alpha_s_max(s_dim, tau, workspace.vars.s, workspace.delta_vars.s);

  if (settings.logging.print_search_direction_logs) {
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

    input.add_Kx_to_y(w, psi, r2, r3, dx, dy, dz, res_x, res_y, res_z);

    for (int i = 0; i < s_dim; ++i) {
      res_s[i] = ds[i] / w[i] + dz[i] + rs[i];
    }

    lin_sys_error = 0.0;

    const int full_dim = x_dim + y_dim + s_dim + s_dim;
    for (int i = 0; i < full_dim; ++i) {
      lin_sys_error = std::max(lin_sys_error, std::fabs(residual[i]));
    }

    print_search_direction_log_header();

    const double quadratic_ms =
        quadratic_model_merit_slope(input, workspace, dx, ds, psi);

    const auto [os_x, os_s, os] =
        get_observed_merit_slope(input, mu, tau, workspace);

    fmt::print(
        fg(fmt::color::green),
        // clang-format off
                   "{:^10} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g}\n",
        // clang-format on
        "", lin_sys_error, alpha_s_max, fixed_dual_ms.total, quadratic_ms, os,
        fixed_dual_ms.x, os_x, fixed_dual_ms.s, os_s, norm(rx, x_dim),
        norm(rs, s_dim));
    const bool suspicious_derivatives =
        (os_x > 0.0 && fixed_dual_ms.x < 0.0) ||
        (os > 0.0 && fixed_dual_ms.total < 0.0) ||
        (std::fabs(fixed_dual_ms.x - os_x) /
             std::max({std::fabs(fixed_dual_ms.x), std::fabs(os_x), 1e-12}) >
         0.5);
    if (settings.logging.print_derivative_check_logs && suspicious_derivatives)
      check_derivatives(input, settings, tau, workspace);
  }

  return std::make_tuple(true, dx, ds, dy, dz, fixed_dual_ms.total,
                         alpha_s_max, dual_residual, max_constraint_violation,
                         max_complementarity, kkt_error, duality_gap,
                         lin_sys_error, num_regularization_increases);
}

auto do_line_search(const Input &input, const Settings &settings,
                    const double mu, const double tau,
                    const double sq_constraint_violation_norm,
                    const double merit_slope, const double alpha_s_max,
                    int &total_ls_iterations, Workspace &workspace)
    -> std::tuple<bool, double, double, double> {
  const int s_dim = input.dimensions.s_dim;
  const int y_dim = input.dimensions.y_dim;

  const auto [m0_f, m0_s, m0_c, m0_g, m0_aug, m0] =
      merit_function(input, workspace, workspace.vars.s, workspace.vars.y,
                     workspace.vars.z, mu);

  bool ls_succeeded = false;
  double alpha =
      settings.line_search.start_ls_with_alpha_s_max ? alpha_s_max : 1.0;
  double trial_alpha = alpha;
  double merit_delta = std::numeric_limits<double>::signaling_NaN();
  double constraint_violation_ratio =
      std::numeric_limits<double>::signaling_NaN();

  if (settings.logging.print_line_search_logs) {
    print_line_search_log_header();
  }

  do {
    trial_alpha = alpha;
    update_next_primal_vars(input, tau, workspace, alpha, true, true);
    const double next_ctc = squared_norm(input.get_c(), y_dim);
    const double next_gsetgse =
        squared_norm(workspace.miscellaneous_workspace.g_plus_s, s_dim);
    const double next_sq_constraint_violation_norm = next_ctc + next_gsetgse;

    // NOTE: the line search is only over the primal variables, so we cannot
    // use next_vars.y and next_vars.z.
    const auto [m_f, m_s, m_c, m_g, m_aug, m] =
        merit_function(input, workspace, workspace.next_vars.s,
                       workspace.vars.y, workspace.vars.z, mu);

    const double dm_f = m_f - m0_f;
    const double dm_s = m_s - m0_s;
    const double dm_c = m_c - m0_c;
    const double dm_g = m_g - m0_g;
    const double dm_aug = m_aug - m0_aug;

    merit_delta = dm_f + dm_s + dm_c + dm_g + dm_aug;

    constraint_violation_ratio = next_sq_constraint_violation_norm /
                                 std::max(sq_constraint_violation_norm, 1e-12);

    if (settings.logging.print_line_search_logs) {
      fmt::print(fg(fmt::color::yellow),
                 // clang-format off
                       "{:^10} {:^+10} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g}\n",
                 // clang-format on
                 "", total_ls_iterations, alpha, m, m_f, std::sqrt(next_ctc),
                 std::sqrt(next_gsetgse), merit_delta, merit_delta / alpha,
                 dm_f, dm_s, dm_c, dm_g, dm_aug);
    }

    ++total_ls_iterations;

    if (merit_slope >
            settings.line_search.min_merit_slope_to_skip_line_search ||
        merit_delta <
            settings.line_search.armijo_factor * merit_slope * alpha) {
      ls_succeeded = true;
      break;
    }

    alpha *= settings.line_search.line_search_factor;
  } while (alpha > settings.line_search.line_search_min_step_size &&
           total_ls_iterations < settings.line_search.max_iterations);

  if (!ls_succeeded) {
    alpha = trial_alpha;
  }

  update_next_dual_vars(input, tau, workspace, alpha);

  return std::make_tuple(ls_succeeded, alpha, m0, constraint_violation_ratio);
}

auto check_settings(const Settings &settings) -> bool {
  const auto is_finite_nonnegative = [](const double value) {
    return std::isfinite(value) && value >= 0.0;
  };
  const auto is_finite_positive = [](const double value) {
    return std::isfinite(value) && value > 0.0;
  };
  const auto is_nonnegative_or_inf = [](const double value) {
    return !std::isnan(value) && value >= 0.0;
  };

  if (settings.max_iterations < 0 || settings.line_search.max_iterations < 0 ||
      settings.num_iterative_refinement_steps < 0) {
    return false;
  }
  if (!is_finite_nonnegative(settings.termination.max_dual_residual) ||
      !is_finite_nonnegative(settings.termination.max_constraint_violation) ||
      !is_finite_nonnegative(settings.termination.max_complementarity_gap) ||
      !is_nonnegative_or_inf(settings.termination.max_duality_gap) ||
      !is_finite_nonnegative(settings.termination.max_cost_change) ||
      !is_finite_nonnegative(settings.termination.max_relative_cost_change) ||
      !is_finite_nonnegative(
          settings.termination.max_suboptimal_constraint_violation) ||
      !is_finite_nonnegative(settings.termination.max_merit_slope)) {
    return false;
  }
  if (!is_finite_positive(settings.line_search.tau) ||
      settings.line_search.tau > 1.0) {
    return false;
  }
  if (!is_finite_nonnegative(settings.barrier.initial_mu) ||
      !is_finite_nonnegative(settings.barrier.mu_update_factor) ||
      settings.barrier.mu_update_factor > 1.0 ||
      !is_finite_nonnegative(settings.barrier.mu_min) ||
      !is_finite_nonnegative(settings.barrier.mu_update_kappa)) {
    return false;
  }
  if (!is_finite_positive(settings.penalty.initial_penalty_parameter) ||
      !is_finite_nonnegative(
          settings.penalty.min_acceptable_constraint_violation_ratio) ||
      !is_finite_positive(settings.penalty.penalty_parameter_increase_factor) ||
      settings.penalty.penalty_parameter_increase_factor < 1.0 ||
      !is_finite_positive(settings.penalty.penalty_parameter_decrease_factor) ||
      settings.penalty.penalty_parameter_decrease_factor > 1.0 ||
      !is_finite_positive(settings.penalty.max_penalty_parameter) ||
      settings.penalty.max_penalty_parameter <
          settings.penalty.initial_penalty_parameter) {
    return false;
  }
  if (!is_finite_nonnegative(settings.line_search.armijo_factor) ||
      settings.line_search.armijo_factor > 1.0 ||
      !is_finite_positive(settings.line_search.line_search_factor) ||
      settings.line_search.line_search_factor >= 1.0 ||
      !is_finite_positive(settings.line_search.line_search_min_step_size) ||
      !std::isfinite(
          settings.line_search.min_merit_slope_to_skip_line_search)) {
    return false;
  }
  const auto &regularization = settings.regularization;
  if (!is_finite_nonnegative(regularization.initial) ||
      regularization.initial > regularization.maximum ||
      !is_finite_positive(regularization.first_positive) ||
      !is_finite_positive(regularization.maximum) ||
      regularization.maximum < regularization.first_positive ||
      regularization.max_attempts <= 0 ||
      !is_finite_positive(regularization.increase_factor) ||
      regularization.increase_factor <= 1.0 ||
      !is_finite_positive(regularization.decrease_factor) ||
      regularization.decrease_factor > 1.0) {
    return false;
  }
  if (settings.logging.print_line_search_logs && !settings.logging.print_logs) {
    return false;
  }
  if (settings.logging.print_search_direction_logs &&
      !settings.logging.print_logs) {
    return false;
  }
  if (settings.logging.print_derivative_check_logs &&
      !settings.logging.print_logs) {
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

    input.model_callback(mci);
  }

  if (!check_settings(settings)) {
    if (settings.assert_checks_pass) {
      assert(false && "check_settings returned false.");
    } else {
      return Output{
          .exit_status = Status::FAILED_CHECK,
          .num_iterations = 0,
          .num_ls_iterations = 0,
          .max_primal_violation = std::numeric_limits<double>::signaling_NaN(),
          .max_dual_violation = -std::numeric_limits<double>::signaling_NaN(),
      };
    }
  }

  const int x_dim = input.dimensions.x_dim;
  const int s_dim = input.dimensions.s_dim;
  const int y_dim = input.dimensions.y_dim;

  for (int i = 0; i < s_dim; ++i) {
    if (workspace.vars.s[i] <= 0.0) {
      if (settings.assert_checks_pass) {
        assert(false && "workspace.vars.s[i] <= 0.0.");
      } else {
        return Output{
            .exit_status = Status::FAILED_CHECK,
            .num_iterations = 0,
            .num_ls_iterations = 0,
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
            .num_ls_iterations = 0,
            .max_primal_violation =
                std::numeric_limits<double>::signaling_NaN(),
            .max_dual_violation = -std::numeric_limits<double>::signaling_NaN(),
        };
      }
    }
  }

  add(input.get_g(), workspace.vars.s, s_dim,
      workspace.miscellaneous_workspace.g_plus_s);

  if (!settings.penalty.warm_start_penalties) {
    std::fill_n(workspace.penalties.y, y_dim,
                settings.penalty.initial_penalty_parameter);
    std::fill_n(workspace.penalties.z, s_dim,
                settings.penalty.initial_penalty_parameter);
  }

  double mu = settings.barrier.initial_mu;
  double psi = settings.regularization.initial;
  const double tau = settings.line_search.tau;

  int total_ls_iterations = 0;
  std::optional<double> previous_cost;

  for (int iteration = 0; iteration < settings.max_iterations; ++iteration) {
    const double f0 = input.get_f();

    const double ctc = squared_norm(input.get_c(), y_dim);
    const double gsetgse =
        squared_norm(workspace.miscellaneous_workspace.g_plus_s, s_dim);

    const double sq_constraint_violation_norm = ctc + gsetgse;

    const auto [factorization_ok, dx, ds, dy, dz, merit_slope, alpha_s_max,
                dual_residual, max_constraint_violation, max_complementarity,
                kkt_error, duality_gap, lin_sys_error,
                num_regularization_increases] =
        compute_search_direction(input, settings, mu, psi, tau, workspace);

    if (!factorization_ok) {
      return Output{
          .exit_status = Status::FACTORIZATION_FAILURE,
          .num_iterations = iteration,
          .num_ls_iterations = total_ls_iterations,
          .max_primal_violation = max_primal_violation(input),
          .max_dual_violation = std::numeric_limits<double>::signaling_NaN(),
      };
    }

    const TerminationChecks termination = check_termination(
        settings, previous_cost, f0, merit_slope, dual_residual,
        max_constraint_violation, max_complementarity, duality_gap);

    const bool hit_ls_iteration_limit =
        total_ls_iterations >= settings.line_search.max_iterations;

    if (settings.logging.print_logs &&
        (termination.solved || hit_ls_iteration_limit)) {
      if (iteration == 0 || settings.logging.print_line_search_logs ||
          settings.logging.print_search_direction_logs) {
        print_log_header();
      }
      const double eta = mean_penalty_parameter(workspace, s_dim, y_dim);
      fmt::print(
          fg(fmt::color::red),
          // clang-format off
                       "{:^+10} {:^10} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^10} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^10} {:^+10.4g} {:^+10.4g} {:^+10.4g}\n",
          // clang-format on
          iteration, "", input.get_f(), std::sqrt(ctc), std::sqrt(gsetgse), "",
          norm(dx, x_dim), norm(ds, s_dim), norm(dy, y_dim), norm(dz, s_dim),
          mu, eta, tau, psi, num_regularization_increases, max_complementarity,
          dual_residual, kkt_error);
    }

    if (termination.solved || termination.stalled) {
      const bool suboptimal =
          max_constraint_violation <
          settings.termination.max_suboptimal_constraint_violation;

      return Output{
          .exit_status = termination.solved
                             ? Status::SOLVED
                             : (suboptimal ? Status::SUBOPTIMAL
                                           : Status::LOCALLY_INFEASIBLE),
          .num_iterations = iteration,
          .num_ls_iterations = total_ls_iterations,
          .max_primal_violation = max_primal_violation(input),
          .max_dual_violation = inf_norm(workspace.nrhs.x, x_dim),
      };
    }

    if (termination.advance_barrier) {
      const double next_mu =
          std::max(mu * settings.barrier.mu_update_factor,
                   settings.barrier.mu_min);
      if (next_mu < mu) {
        mu = next_mu;
        continue;
      }
    }

    if (hit_ls_iteration_limit) {
      return Output{
          .exit_status = Status::LINE_SEARCH_ITERATION_LIMIT,
          .num_iterations = iteration,
          .num_ls_iterations = total_ls_iterations,
          .max_primal_violation = max_primal_violation(input),
          .max_dual_violation = inf_norm(workspace.nrhs.x, x_dim),
      };
    }

    bool ls_succeeded;
    double alpha, m0;

    if (settings.line_search.skip_line_search) {
      alpha = alpha_s_max;
      update_next_primal_vars(input, tau, workspace, alpha, true, true);
      update_next_dual_vars(input, tau, workspace, alpha);
      ls_succeeded = true;
      m0 = 0.0;
    } else {
      std::tie(ls_succeeded, alpha, m0, std::ignore) = do_line_search(
          input, settings, mu, tau, sq_constraint_violation_norm, merit_slope,
          alpha_s_max, total_ls_iterations, workspace);
    }

    if (settings.logging.print_logs) {
      if (iteration == 0 || settings.logging.print_line_search_logs) {
        print_log_header();
      }

      const double eta = mean_penalty_parameter(workspace, s_dim, y_dim);
      fmt::print(
          fg(fmt::color::red),
          // clang-format off
                       "{:^+10} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g}  {:^+10.4g} {:^+10.4g} {:^10} {:^+10.4g} {:^+10.4g} {:^+10.4g}\n",
          // clang-format on
          iteration, alpha, f0, std::sqrt(ctc), std::sqrt(gsetgse), m0,
          norm(dx, x_dim), norm(ds, s_dim), norm(dy, y_dim), norm(dz, s_dim),
          mu, eta, tau, psi, num_regularization_increases, max_complementarity,
          dual_residual, kkt_error);
    }

    if (settings.line_search.enable_line_search_failures && !ls_succeeded) {
      return Output{
          .exit_status = Status::LINE_SEARCH_FAILURE,
          .num_iterations = iteration,
          .num_ls_iterations = total_ls_iterations,
          .max_primal_violation = max_primal_violation(input),
          .max_dual_violation = inf_norm(workspace.nrhs.x, x_dim),
      };
    }

    std::swap(workspace.vars, workspace.next_vars);
    previous_cost = f0;

    if (input.timeout_callback()) {
      return Output{
          .exit_status = Status::TIMEOUT,
          .num_iterations = iteration,
          .num_ls_iterations = total_ls_iterations,
          .max_primal_violation = max_primal_violation(input),
          .max_dual_violation = inf_norm(workspace.nrhs.x, x_dim),
      };
    }

    if (kkt_error <= settings.barrier.mu_update_kappa * mu) {
      mu = std::max(mu * settings.barrier.mu_update_factor,
                    settings.barrier.mu_min);
    }
    psi = decreased_regularization(settings, psi);

    const bool any_penalty_increased =
        update_penalty_parameters(input, settings, workspace);
    if (any_penalty_increased) {
      // Reset regularization when penalty increases, to stabilize the
      // modified KKT system.
      psi = std::max(psi, settings.regularization.initial);
    }
  }

  return Output{
      .exit_status = Status::ITERATION_LIMIT,
      .num_iterations = settings.max_iterations,
      .num_ls_iterations = total_ls_iterations,
      .max_primal_violation = max_primal_violation(input),
      .max_dual_violation = inf_norm(workspace.nrhs.x, x_dim),
  };
}

} // namespace sip
