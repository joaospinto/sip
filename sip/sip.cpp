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

auto unscaled_max_abs(const double *values, const double *scaling,
                      const int size) -> double {
  if (scaling == nullptr) {
    return max_abs_or_inf(values, size);
  }
  double result = 0.0;
  for (int i = 0; i < size; ++i) {
    const double value = std::fabs(values[i] / scaling[i]);
    result = std::isnan(value) ? std::numeric_limits<double>::infinity()
                               : std::max(result, value);
  }
  return result;
}

auto unscaled_max_positive(const double *values, const double *scaling,
                           const int size) -> double {
  if (scaling == nullptr) {
    return max_positive_or_inf(values, size);
  }
  double result = 0.0;
  for (int i = 0; i < size; ++i) {
    const double value = values[i] / scaling[i];
    result = std::isnan(value) ? std::numeric_limits<double>::infinity()
                               : std::max(result, value);
  }
  return result;
}

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
  return std::max(
      unscaled_max_abs(input.get_c(), input.residual_scaling.equality, y_dim),
      unscaled_max_positive(input.get_g(), input.residual_scaling.inequality,
                            s_dim));
}

auto mean_complementarity(const Workspace &workspace, const int s_dim)
    -> double {
  if (s_dim == 0) {
    return 0.0;
  }
  return dot(workspace.vars.s, workspace.vars.z, s_dim) / s_dim;
}

auto unscaled_max_regularized_difference(const double *lhs, const double *rhs,
                                         const double *regularization,
                                         const double *scaling, const int size)
    -> double {
  double result = 0.0;
  for (int i = 0; i < size; ++i) {
    const double value = regularization[i] * (lhs[i] - rhs[i]);
    result = std::max(
        result, std::fabs(scaling == nullptr ? value : value / scaling[i]));
  }
  return result;
}

auto unscaled_max_regularized_difference(const double *lhs, const double *rhs,
                                         const double regularization,
                                         const double *scaling, const int size)
    -> double {
  double result = 0.0;
  for (int i = 0; i < size; ++i) {
    const double value = regularization * (lhs[i] - rhs[i]);
    result = std::max(
        result, std::fabs(scaling == nullptr ? value : value / scaling[i]));
  }
  return result;
}

auto unscaled_max_inverse_regularized_difference(
    const double *lhs, const double *rhs, const double *inverse_regularization,
    const double *scaling, const int size) -> double {
  double result = 0.0;
  for (int i = 0; i < size; ++i) {
    const double value = (lhs[i] - rhs[i]) / inverse_regularization[i];
    result = std::max(
        result, std::fabs(scaling == nullptr ? value : value / scaling[i]));
  }
  return result;
}

auto unregularized_residuals(const Input &input, Workspace &workspace)
    -> std::pair<double, double> {
  const int x_dim = input.dimensions.x_dim;
  const int s_dim = input.dimensions.s_dim;
  const int y_dim = input.dimensions.y_dim;

  double *dual = workspace.nrhs.x;
  std::copy_n(input.get_grad_f(), x_dim, dual);
  input.add_CTx_to_y(workspace.vars.y, dual);
  input.add_GTx_to_y(workspace.vars.z, dual);

  const double primal = std::max(
      unscaled_max_abs(input.get_c(), input.residual_scaling.equality, y_dim),
      unscaled_max_abs(workspace.miscellaneous_workspace.g_plus_s,
                       input.residual_scaling.inequality, s_dim));
  return {primal, unscaled_max_abs(dual, input.residual_scaling.dual, x_dim)};
}

auto merit_function(const Input &input, const Settings &settings,
                    const Workspace &workspace, const double *x,
                    const double *s, const double *y, const double *z,
                    const double mu, const double psi)
    -> std::tuple<double, double, double, double, double, double> {
  const int x_dim = input.dimensions.x_dim;
  const int s_dim = input.dimensions.s_dim;
  const int y_dim = input.dimensions.y_dim;
  const double *c = input.get_c();
  const double *gps = workspace.miscellaneous_workspace.g_plus_s;
  const double s_term = -mu * sum_of_logs(s, s_dim);
  double c_term = 0.0;
  double g_term = 0.0;
  double aug_term = 0.0;

  if (settings.proximal.use_dual_center) {
    for (int i = 0; i < y_dim; ++i) {
      const double eta = workspace.penalties.y[i];
      const double regularized_residual =
          c[i] - (y[i] - workspace.proximal_centers.y[i]) / eta;
      c_term += workspace.proximal_centers.y[i] * c[i];
      aug_term += 0.5 * eta *
                  (c[i] * c[i] + regularized_residual * regularized_residual);
    }
    for (int i = 0; i < s_dim; ++i) {
      const double eta = workspace.penalties.z[i];
      if (workspace.csd_workspace.r3[i] == 0.0) {
        g_term += z[i] * gps[i];
        aug_term += 0.5 * eta * gps[i] * gps[i];
        continue;
      }
      const double regularized_residual =
          gps[i] - (z[i] - workspace.proximal_centers.z[i]) / eta;
      g_term += workspace.proximal_centers.z[i] * gps[i];
      aug_term +=
          0.5 * eta *
          (gps[i] * gps[i] + regularized_residual * regularized_residual);
    }
  } else {
    c_term = dot(c, y, y_dim);
    g_term = dot(gps, z, s_dim);
    aug_term = 0.5 * (weighted_squared_norm(c, workspace.penalties.y, y_dim) +
                      weighted_squared_norm(gps, workspace.penalties.z, s_dim));
  }

  if (settings.proximal.use_primal_center) {
    for (int i = 0; i < x_dim; ++i) {
      const double displacement = x[i] - workspace.proximal_centers.x[i];
      aug_term += 0.5 * psi * displacement * displacement;
    }
  }

  const double barrier_lagrangian = input.get_f() + s_term + c_term + g_term;
  const double merit = barrier_lagrangian + aug_term;
  return std::make_tuple(input.get_f(), s_term, c_term, g_term, aug_term,
                         merit);
}

auto filter_accepts(const FilterWorkspace &filter, const Settings &settings,
                    const double theta, const double f) -> bool {
  for (int i = 0; i < filter.size; ++i) {
    if (theta >
            (1.0 - settings.line_search.filter_gamma_theta) * filter.theta[i] &&
        f > filter.f[i] -
                settings.line_search.filter_gamma_f * filter.theta[i]) {
      return false;
    }
  }
  return true;
}

void add_filter_entry(FilterWorkspace &filter, const Settings &settings,
                      const double theta, const double f) {
  if (!filter_accepts(filter, settings, theta, f)) {
    return;
  }

  int next_size = 0;
  for (int i = 0; i < filter.size; ++i) {
    const bool dominated =
        filter.theta[i] >=
            (1.0 - settings.line_search.filter_gamma_theta) * theta &&
        filter.f[i] >= f - settings.line_search.filter_gamma_f * theta;
    if (!dominated) {
      filter.theta[next_size] = filter.theta[i];
      filter.f[next_size] = filter.f[i];
      ++next_size;
    }
  }
  assert(next_size < filter.capacity);
  filter.theta[next_size] = theta;
  filter.f[next_size] = f;
  filter.size = next_size + 1;
}

auto get_fraction_to_boundary_step(const int dim, const double tau,
                                   const double *value, const double *direction)
    -> double {
  // value + alpha * direction >= (1 - tau) * value

  double alpha = 1.0;

  for (int i = 0; i < dim; ++i) {
    if (direction[i] < 0.0) {
      alpha = std::min(alpha, tau * value[i] / -direction[i]);
    }
  }

  return alpha;
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
  if (decreased >= regularization.first_positive) {
    return decreased;
  }
  return settings.proximal.use_primal_center ? regularization.first_positive
                                             : 0.0;
}

struct TerminationChecks {
  bool solved;
  bool stalled;
  bool advance_barrier;
};

struct ProximalCenterUpdateRejections {
  void reject(const double current_residual) {
    if (count == 0) {
      residual_at_start = current_residual;
    }
    ++count;
  }

  void reset() {
    count = 0;
    residual_at_start = std::numeric_limits<double>::infinity();
  }

  auto has_reduced_residual(const double residual,
                            const double reduction_factor) const -> bool {
    return residual < reduction_factor * residual_at_start;
  }

  int count{0};
  double residual_at_start{std::numeric_limits<double>::infinity()};
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
  const bool advance_barrier =
      merit_slope_too_small && primal_feasibility_satisfied &&
      dual_residual_satisfied && !complementarity_satisfied;

  return TerminationChecks{
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

auto update_initial_penalties_for_linearized_constraint_violation(
    const Input &input, const Settings &settings,
    double &previous_linearized_violation_ratio, Workspace &workspace) -> bool {
  const int s_dim = input.dimensions.s_dim;
  const int y_dim = input.dimensions.y_dim;
  const double min_ratio =
      settings.penalty.min_acceptable_constraint_violation_ratio;
  const double increase_factor =
      settings.penalty.penalty_parameter_increase_factor;
  const double max_penalty = settings.penalty.max_penalty_parameter;
  const double max_violation = settings.termination.max_constraint_violation;
  const bool currently_feasible =
      unscaled_max_abs(input.get_c(), input.residual_scaling.equality, y_dim) <=
          max_violation &&
      unscaled_max_abs(workspace.miscellaneous_workspace.g_plus_s,
                       input.residual_scaling.inequality,
                       s_dim) <= max_violation;
  if (currently_feasible) {
    return false;
  }

  double *linearized_c = workspace.next_vars.y;
  std::copy_n(input.get_c(), y_dim, linearized_c);
  input.add_Cx_to_y(workspace.delta_vars.x, linearized_c);

  double *linearized_gps = workspace.next_vars.s;
  std::copy_n(workspace.miscellaneous_workspace.g_plus_s, s_dim,
              linearized_gps);
  input.add_Gx_to_y(workspace.delta_vars.x, linearized_gps);
  for (int i = 0; i < s_dim; ++i) {
    linearized_gps[i] += workspace.delta_vars.s[i];
  }

  const auto violation_ratio = [](const double current, const double linearized,
                                  const double acceptable_violation) {
    const double reference = std::max(std::fabs(current), acceptable_violation);
    if (reference > 0.0) {
      return std::fabs(linearized) / reference;
    }
    return linearized == 0.0 ? 0.0 : std::numeric_limits<double>::infinity();
  };
  const auto increase_if_needed = [&](const double current,
                                      const double linearized,
                                      const double acceptable_violation,
                                      double &penalty) {
    if (std::fabs(linearized) <= acceptable_violation) {
      return false;
    }
    const double squared_current = current * current;
    if (squared_current > acceptable_violation * acceptable_violation &&
        linearized * linearized <= min_ratio * squared_current) {
      return false;
    }
    const double increased = std::min(penalty * increase_factor, max_penalty);
    const bool changed = increased > penalty;
    penalty = increased;
    return changed;
  };

  double max_linearized_violation_ratio = 0.0;
  for (int i = 0; i < y_dim; ++i) {
    const double scale = input.residual_scaling.equality == nullptr
                             ? 1.0
                             : input.residual_scaling.equality[i];
    max_linearized_violation_ratio =
        std::max(max_linearized_violation_ratio,
                 violation_ratio(input.get_c()[i], linearized_c[i],
                                 max_violation * scale));
  }
  for (int i = 0; i < s_dim; ++i) {
    const double scale = input.residual_scaling.inequality == nullptr
                             ? 1.0
                             : input.residual_scaling.inequality[i];
    max_linearized_violation_ratio =
        std::max(max_linearized_violation_ratio,
                 violation_ratio(workspace.miscellaneous_workspace.g_plus_s[i],
                                 linearized_gps[i], max_violation * scale));
  }
  if (max_linearized_violation_ratio >= previous_linearized_violation_ratio) {
    return false;
  }
  previous_linearized_violation_ratio = max_linearized_violation_ratio;

  bool any_increased = false;
  for (int i = 0; i < y_dim; ++i) {
    const double scale = input.residual_scaling.equality == nullptr
                             ? 1.0
                             : input.residual_scaling.equality[i];
    any_increased |=
        increase_if_needed(input.get_c()[i], linearized_c[i],
                           max_violation * scale, workspace.penalties.y[i]);
  }
  for (int i = 0; i < s_dim; ++i) {
    const double scale = input.residual_scaling.inequality == nullptr
                             ? 1.0
                             : input.residual_scaling.inequality[i];
    any_increased |= increase_if_needed(
        workspace.miscellaneous_workspace.g_plus_s[i], linearized_gps[i],
        max_violation * scale, workspace.penalties.z[i]);
  }
  return any_increased;
}

void decrease_dual_regularization(const Input &input, Workspace &workspace,
                                  const double max_penalty_parameter,
                                  const double reduction_factor) {
  const int s_dim = input.dimensions.s_dim;
  const int y_dim = input.dimensions.y_dim;
  const double multiplier = reduction_factor > 0.0
                                ? 1.0 / reduction_factor
                                : std::numeric_limits<double>::infinity();
  for (int i = 0; i < y_dim; ++i) {
    workspace.penalties.y[i] =
        std::min(max_penalty_parameter, multiplier * workspace.penalties.y[i]);
  }
  for (int i = 0; i < s_dim; ++i) {
    workspace.penalties.z[i] =
        std::min(max_penalty_parameter, multiplier * workspace.penalties.z[i]);
  }
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

void update_next_dual_vars(const Input &input, const Settings &settings,
                           const double tau, Workspace &workspace,
                           const double dual_step) {
  const int s_dim = input.dimensions.s_dim;
  const int y_dim = input.dimensions.y_dim;

  for (int i = 0; i < y_dim; ++i) {
    workspace.next_vars.y[i] =
        workspace.vars.y[i] + dual_step * workspace.delta_vars.y[i];
  }

  for (int i = 0; i < s_dim; ++i) {
    const double next_z =
        workspace.vars.z[i] + dual_step * workspace.delta_vars.z[i];
    workspace.next_vars.z[i] =
        settings.barrier.use_predictor_corrector ||
                settings.proximal.use_dual_center
            ? next_z
            : std::max(next_z, (1.0 - tau) * workspace.vars.z[i]);
    assert(workspace.next_vars.z[i] > 0.0);
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

auto get_observed_merit_slope(const Input &input, const Settings &settings,
                              const double mu, const double psi,
                              const double tau, Workspace &workspace)
    -> std::tuple<double, double, double> {
  const auto compute_empirical_merit_slope = [&](const bool update_x,
                                                 const bool update_s,
                                                 const bool update_dual) {
    const auto get_perturbed_merit = [&](const double beta) -> double {
      update_next_primal_vars(input, tau, workspace, beta, update_x, update_s);
      if (update_dual) {
        update_next_dual_vars(input, settings, tau, workspace, beta);
      }
      const double *y = update_dual ? workspace.next_vars.y : workspace.vars.y;
      const double *z = update_dual ? workspace.next_vars.z : workspace.vars.z;
      const auto [_mP_f, _mP_s, _mP_c, _mP_g, _mP_aug, mP] =
          merit_function(input, settings, workspace, workspace.next_vars.x,
                         workspace.next_vars.s, y, z, mu, psi);
      return mP;
    };

    const double h = std::sqrt(std::numeric_limits<double>::epsilon());

    const double mP = get_perturbed_merit(h);
    const double mM = get_perturbed_merit(-h);

    return (mP - mM) / (2 * h);
  };

  const double os_x = compute_empirical_merit_slope(true, false, false);
  const double os_s = compute_empirical_merit_slope(false, true, false);
  const double os = compute_empirical_merit_slope(
      true, true, settings.proximal.use_dual_center);

  update_next_primal_vars(input, tau, workspace, 0.0, false, false);
  update_next_dual_vars(input, settings, tau, workspace, 0.0);

  return std::make_tuple(os_x, os_s, os);
}

struct MeritSlope {
  double x;
  double s;
  double dual;
  double total;
};

auto merit_slope(const Input &input, const Workspace &workspace,
                 const Settings &settings, const double psi, const double mu,
                 const double *dx, const double *ds, const double *dy,
                 const double *dz) -> MeritSlope {
  const int x_dim = input.dimensions.x_dim;
  const int s_dim = input.dimensions.s_dim;
  const int y_dim = input.dimensions.y_dim;
  const double *c = input.get_c();
  const double *gps = workspace.miscellaneous_workspace.g_plus_s;

  if (settings.proximal.use_dual_center) {
    double *gradient_x = workspace.next_vars.x;
    double *multiplier_y = workspace.next_vars.y;
    double *multiplier_z = workspace.next_vars.z;

    std::copy_n(input.get_grad_f(), x_dim, gradient_x);
    double dual_slope = 0.0;
    for (int i = 0; i < y_dim; ++i) {
      const double eta = workspace.penalties.y[i];
      const double regularized_residual =
          c[i] - (workspace.vars.y[i] - workspace.proximal_centers.y[i]) / eta;
      multiplier_y[i] = 2.0 * (workspace.proximal_centers.y[i] + eta * c[i]) -
                        workspace.vars.y[i];
      dual_slope -= regularized_residual * dy[i];
    }
    for (int i = 0; i < s_dim; ++i) {
      const double eta = workspace.penalties.z[i];
      if (workspace.csd_workspace.r3[i] == 0.0) {
        multiplier_z[i] = workspace.vars.z[i] + eta * gps[i];
        dual_slope += gps[i] * dz[i];
        continue;
      }
      const double regularized_residual =
          gps[i] -
          (workspace.vars.z[i] - workspace.proximal_centers.z[i]) / eta;
      multiplier_z[i] = 2.0 * (workspace.proximal_centers.z[i] + eta * gps[i]) -
                        workspace.vars.z[i];
      dual_slope -= regularized_residual * dz[i];
    }
    input.add_CTx_to_y(multiplier_y, gradient_x);
    input.add_GTx_to_y(multiplier_z, gradient_x);
    if (settings.proximal.use_primal_center) {
      for (int i = 0; i < x_dim; ++i) {
        gradient_x[i] +=
            psi * (workspace.vars.x[i] - workspace.proximal_centers.x[i]);
      }
    }

    const double x_slope = dot(gradient_x, dx, x_dim);
    double s_slope = 0.0;
    for (int i = 0; i < s_dim; ++i) {
      s_slope += (multiplier_z[i] - mu / workspace.vars.s[i]) * ds[i];
    }
    return {.x = x_slope,
            .s = s_slope,
            .dual = dual_slope,
            .total = x_slope + s_slope + dual_slope};
  }

  double *tmp_y = workspace.next_vars.y;
  double *tmp_s = workspace.next_vars.s;

  std::fill_n(tmp_y, y_dim, 0.0);
  input.add_Cx_to_y(dx, tmp_y);

  std::fill_n(tmp_s, s_dim, 0.0);
  input.add_Gx_to_y(dx, tmp_s);

  double x_slope = dot(workspace.nrhs.x, dx, x_dim) +
                   weighted_dot(c, workspace.penalties.y, tmp_y, y_dim) +
                   weighted_dot(gps, workspace.penalties.z, tmp_s, s_dim);
  if (settings.proximal.use_primal_center) {
    for (int i = 0; i < x_dim; ++i) {
      x_slope +=
          psi * (workspace.vars.x[i] - workspace.proximal_centers.x[i]) * dx[i];
    }
  }

  double s_slope = weighted_dot(gps, workspace.penalties.z, ds, s_dim);
  for (int i = 0; i < s_dim; ++i) {
    s_slope += (workspace.vars.z[i] - mu / workspace.vars.s[i]) * ds[i];
  }

  return {.x = x_slope, .s = s_slope, .dual = 0.0, .total = x_slope + s_slope};
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
                  double, double, double, double, int> {
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

  const bool use_predictor_corrector = settings.barrier.use_predictor_corrector;
  for (int i = 0; i < s_dim; ++i) {
    const double complementarity_ratio = s[i] / z[i];
    w[i] = use_predictor_corrector
               ? complementarity_ratio
               : std::clamp(complementarity_ratio, 1e-18, 1e18);
  }

  for (int i = 0; i < y_dim; ++i) {
    r2[i] = 1.0 / workspace.penalties.y[i];
  }
  for (int i = 0; i < s_dim; ++i) {
    r3[i] = 1.0 / workspace.penalties.z[i];
  }

  int num_regularization_increases = 0;
  bool factorization_ok = false;
  double numerical_regularization = 0.0;
  for (int attempt = 0; attempt < settings.regularization.max_attempts;
       ++attempt) {
    factorization_ok = input.factor(w, psi, r2, r3, numerical_regularization);
    if (factorization_ok) {
      break;
    }
    if (use_predictor_corrector || settings.proximal.use_primal_center ||
        settings.proximal.use_dual_center) {
      numerical_regularization = numerical_regularization == 0.0
                                     ? 1e-14
                                     : 10.0 * numerical_regularization;
      ++num_regularization_increases;
      continue;
    }
    const double next_psi = increased_regularization(settings, psi);
    if (next_psi <= psi || next_psi > settings.regularization.maximum) {
      break;
    }
    psi = next_psi;
    ++num_regularization_increases;
  }

  if (!factorization_ok) {
    return std::make_tuple(false, dx, ds, dy, dz, 0.0, 0.0, 0.0,
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

  const auto solve_direction = [&](const double target_mu,
                                   const double *affine_ds,
                                   const double *affine_dz) {
    std::copy_n(rx, x_dim, bx);
    std::copy_n(c, y_dim, by);
    if (settings.proximal.use_primal_center) {
      for (int i = 0; i < x_dim; ++i) {
        bx[i] += psi * (workspace.vars.x[i] - workspace.proximal_centers.x[i]);
      }
    }
    if (settings.proximal.use_dual_center) {
      for (int i = 0; i < y_dim; ++i) {
        by[i] +=
            r2[i] * (workspace.proximal_centers.y[i] - workspace.vars.y[i]);
      }
    }
    for (int i = 0; i < s_dim; ++i) {
      const double second_order =
          affine_ds == nullptr ? 0.0 : affine_ds[i] * affine_dz[i];
      rs[i] = z[i] - target_mu / s[i] + second_order / s[i];
      bz[i] = rz[i] - w[i] * rs[i];
      if (settings.proximal.use_dual_center && r3[i] != 0.0) {
        bz[i] +=
            r3[i] * (workspace.proximal_centers.z[i] - workspace.vars.z[i]);
      }
    }
    for (int i = 0; i < dim_3x3; ++i) {
      b[i] = -b[i];
    }

    input.solve(b, v);
    const auto compute_refinement_residual = [&]() {
      // res = Kv - b; Ku = res; therefore K(v - u) = b.
      double *res_x = residual;
      double *res_y = res_x + x_dim;
      double *res_z = res_y + y_dim;
      for (int i = 0; i < dim_3x3; ++i) {
        residual[i] = -b[i];
      }
      input.add_Kx_to_y(w, psi, r2, r3, vx, vy, vz, res_x, res_y, res_z);
    };
    compute_refinement_residual();
    double previous_refinement_error = max_abs_or_inf(residual, dim_3x3);
    const int num_iterative_refinement_steps =
        settings.num_iterative_refinement_steps;
    for (int j = 0; j < num_iterative_refinement_steps; ++j) {
      input.solve(residual, u);
      if (max_abs_or_inf(u, dim_3x3) >
          10.0 * std::max(1.0, max_abs_or_inf(v, dim_3x3))) {
        break;
      }
      for (int i = 0; i < dim_3x3; ++i) {
        v[i] -= u[i];
      }
      compute_refinement_residual();
      const double refinement_error = max_abs_or_inf(residual, dim_3x3);
      const double improvement = previous_refinement_error / refinement_error;
      if (improvement < 5.0) {
        if (improvement <= 1.0) {
          for (int i = 0; i < dim_3x3; ++i) {
            v[i] += u[i];
          }
        }
        break;
      }
      previous_refinement_error = refinement_error;
    }

    const double *solution = v;
    std::copy_n(solution, x_dim, dx);
    solution += x_dim;
    std::copy_n(solution, y_dim, dy);
    solution += y_dim;
    std::copy_n(solution, s_dim, dz);
    for (int i = 0; i < s_dim; ++i) {
      ds[i] = -w[i] * (dz[i] + rs[i]);
    }
  };

  double stationarity_merit_slope;
  if (settings.barrier.use_predictor_corrector && s_dim > 0) {
    solve_direction(mu, nullptr, nullptr);
    stationarity_merit_slope =
        merit_slope(input, workspace, settings, psi, mu, dx, ds, dy, dz).total;

    solve_direction(0.0, nullptr, nullptr);
    double *affine_ds = workspace.next_vars.s;
    double *affine_dz = workspace.next_vars.z;
    std::copy_n(ds, s_dim, affine_ds);
    std::copy_n(dz, s_dim, affine_dz);

    const double alpha_s =
        get_fraction_to_boundary_step(s_dim, tau, s, affine_ds);
    const double alpha_z =
        get_fraction_to_boundary_step(s_dim, tau, z, affine_dz);
    double current_mu = 0.0;
    double affine_mu = 0.0;
    for (int i = 0; i < s_dim; ++i) {
      current_mu += s[i] * z[i];
      affine_mu +=
          (s[i] + alpha_s * affine_ds[i]) * (z[i] + alpha_z * affine_dz[i]);
    }
    current_mu /= s_dim;
    affine_mu /= s_dim;
    const double mu_ratio =
        current_mu > 0.0 ? std::clamp(affine_mu / current_mu, 0.0, 1.0) : 0.0;
    const double sigma = mu_ratio * mu_ratio * mu_ratio;
    solve_direction(sigma * current_mu, affine_ds, affine_dz);
  } else {
    solve_direction(mu, nullptr, nullptr);
    stationarity_merit_slope =
        merit_slope(input, workspace, settings, psi, mu, dx, ds, dy, dz).total;
  }

  const auto current_merit_slope =
      merit_slope(input, workspace, settings, psi, mu, dx, ds, dy, dz);

  for (int i = 0; i < y_dim; ++i) {
    const double abs_c =
        std::fabs(c[i] / (input.residual_scaling.equality == nullptr
                              ? 1.0
                              : input.residual_scaling.equality[i]));
    max_constraint_violation = std::isnan(abs_c)
                                   ? std::numeric_limits<double>::infinity()
                                   : std::max(max_constraint_violation, abs_c);
  }

  for (int i = 0; i < s_dim; ++i) {
    const double abs_gps =
        std::fabs(workspace.miscellaneous_workspace.g_plus_s[i] /
                  (input.residual_scaling.inequality == nullptr
                       ? 1.0
                       : input.residual_scaling.inequality[i]));
    max_constraint_violation =
        std::isnan(abs_gps) ? std::numeric_limits<double>::infinity()
                            : std::max(max_constraint_violation, abs_gps);
    const double sz = s[i] * z[i];
    max_complementarity = std::isnan(sz)
                              ? std::numeric_limits<double>::infinity()
                              : std::max(max_complementarity, sz);
  }

  for (int i = 0; i < x_dim; ++i) {
    const double abs_rx =
        std::fabs(rx[i] / (input.residual_scaling.dual == nullptr
                               ? 1.0
                               : input.residual_scaling.dual[i]));
    dual_residual = std::isnan(abs_rx) ? std::numeric_limits<double>::infinity()
                                       : std::max(dual_residual, abs_rx);
  }

  kkt_error = dual_residual;
  kkt_error = std::max(kkt_error, max_constraint_violation);
  kkt_error = std::max(kkt_error, max_complementarity);
  if (std::isnan(kkt_error)) {
    kkt_error = std::numeric_limits<double>::infinity();
  }

  const auto alpha_s_max = get_fraction_to_boundary_step(
      s_dim, tau, workspace.vars.s, workspace.delta_vars.s);

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
        get_observed_merit_slope(input, settings, mu, psi, tau, workspace);

    fmt::print(
        fg(fmt::color::green),
        // clang-format off
                   "{:^10} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g}\n",
        // clang-format on
        "", lin_sys_error, alpha_s_max, current_merit_slope.total, quadratic_ms,
        os, current_merit_slope.x, os_x, current_merit_slope.s, os_s,
        norm(rx, x_dim), norm(rs, s_dim));
    const bool suspicious_derivatives =
        (os_x > 0.0 && current_merit_slope.x < 0.0) ||
        (os > 0.0 && current_merit_slope.total < 0.0) ||
        (std::fabs(current_merit_slope.x - os_x) /
             std::max(
                 {std::fabs(current_merit_slope.x), std::fabs(os_x), 1e-12}) >
         0.5);
    if (settings.logging.print_derivative_check_logs && suspicious_derivatives)
      check_derivatives(input, settings, tau, workspace);
  }

  return std::make_tuple(
      true, dx, ds, dy, dz, stationarity_merit_slope, current_merit_slope.total,
      alpha_s_max, dual_residual, max_constraint_violation, max_complementarity,
      kkt_error, duality_gap, lin_sys_error, num_regularization_increases);
}

auto do_line_search(const Input &input, const Settings &settings,
                    const double mu, const double psi, const double tau,
                    const double sq_constraint_violation_norm,
                    const double merit_slope, const double alpha_s_max,
                    const double alpha_z_max, int &total_ls_iterations,
                    FilterWorkspace &filter, Workspace &workspace)
    -> std::tuple<bool, double, double, double> {
  const int s_dim = input.dimensions.s_dim;
  const int y_dim = input.dimensions.y_dim;

  const auto [m0_f, m0_s, m0_c, m0_g, m0_aug, m0] = merit_function(
      input, settings, workspace, workspace.vars.x, workspace.vars.s,
      workspace.vars.y, workspace.vars.z, mu, psi);

  bool ls_succeeded = false;
  double alpha =
      settings.line_search.start_ls_with_alpha_s_max ? alpha_s_max : 1.0;
  if (settings.proximal.use_dual_center) {
    alpha = std::min(alpha, alpha_z_max);
  }
  double trial_alpha = alpha;
  double merit_delta = std::numeric_limits<double>::signaling_NaN();
  double constraint_violation_ratio =
      std::numeric_limits<double>::signaling_NaN();
  bool accepted_without_filter = false;
  const bool filter_active =
      settings.line_search.use_filter_line_search &&
      total_ls_iterations >=
          settings.line_search.filter_min_total_line_search_iterations;

  if (settings.logging.print_line_search_logs) {
    print_line_search_log_header();
  }

  do {
    trial_alpha = alpha;
    update_next_primal_vars(input, tau, workspace, alpha, true, true);
    if (settings.proximal.use_dual_center) {
      update_next_dual_vars(input, settings, tau, workspace, alpha);
    }
    const double next_ctc = squared_norm(input.get_c(), y_dim);
    const double next_gsetgse =
        squared_norm(workspace.miscellaneous_workspace.g_plus_s, s_dim);
    const double next_sq_constraint_violation_norm = next_ctc + next_gsetgse;
    const double next_theta = std::sqrt(next_sq_constraint_violation_norm);

    const double *trial_y = settings.proximal.use_dual_center
                                ? workspace.next_vars.y
                                : workspace.vars.y;
    const double *trial_z = settings.proximal.use_dual_center
                                ? workspace.next_vars.z
                                : workspace.vars.z;
    const auto [m_f, m_s, m_c, m_g, m_aug, m] =
        merit_function(input, settings, workspace, workspace.next_vars.x,
                       workspace.next_vars.s, trial_y, trial_z, mu, psi);

    const double dm_f = m_f - m0_f;
    const double dm_s = m_s - m0_s;
    const double dm_c = m_c - m0_c;
    const double dm_g = m_g - m0_g;
    const double dm_aug = m_aug - m0_aug;

    merit_delta = dm_f + dm_s + dm_c + dm_g + dm_aug;

    constraint_violation_ratio = next_sq_constraint_violation_norm /
                                 std::max(sq_constraint_violation_norm, 1e-12);
    const double current_theta = std::sqrt(sq_constraint_violation_norm);
    const bool acceptable_to_current_iterate =
        next_theta <=
            (1.0 - settings.line_search.filter_gamma_theta) * current_theta ||
        m_f <= m0_f - settings.line_search.filter_gamma_f * current_theta;
    const bool filter_accept =
        filter_active && acceptable_to_current_iterate &&
        filter_accepts(filter, settings, next_theta, m_f);
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

    if (settings.proximal.use_dual_center) {
      const bool skip_line_search =
          merit_slope <= 0.0 &&
          merit_slope >
              settings.line_search.min_merit_slope_to_skip_line_search;
      accepted_without_filter =
          skip_line_search ||
          (merit_slope < 0.0 &&
           merit_delta <
               settings.line_search.armijo_factor * merit_slope * alpha);
    } else {
      accepted_without_filter =
          merit_slope >
              settings.line_search.min_merit_slope_to_skip_line_search ||
          merit_delta <
              settings.line_search.armijo_factor * merit_slope * alpha;
    }
    if (accepted_without_filter || filter_accept) {
      ls_succeeded = true;
      break;
    }

    alpha *= settings.line_search.line_search_factor;
  } while (alpha > settings.line_search.line_search_min_step_size &&
           total_ls_iterations < settings.line_search.max_iterations);

  if (!ls_succeeded) {
    alpha = trial_alpha;
  }

  const double dual_step = settings.proximal.use_dual_center
                               ? alpha
                               : (settings.barrier.use_predictor_corrector
                                      ? std::min(alpha, alpha_z_max)
                                      : alpha);
  update_next_dual_vars(input, settings, tau, workspace, dual_step);

  if (ls_succeeded && filter_active && !accepted_without_filter) {
    add_filter_entry(filter, settings, std::sqrt(sq_constraint_violation_norm),
                     m0_f);
  }

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

  if (settings.barrier.use_predictor_corrector &&
      !settings.line_search.skip_line_search) {
    return false;
  }

  if (settings.max_iterations < 0 || settings.line_search.max_iterations < 0 ||
      settings.line_search.filter_min_total_line_search_iterations < 0 ||
      settings.num_iterative_refinement_steps < 0 ||
      settings.proximal.max_consecutive_center_update_rejections <= 0 ||
      settings.termination
              .num_consecutive_stalled_iterations_before_termination <= 0) {
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
  if (!is_finite_positive(
          settings.proximal.initial_continuation_regularization_floor)) {
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
      !is_finite_nonnegative(settings.line_search.filter_gamma_theta) ||
      settings.line_search.filter_gamma_theta > 1.0 ||
      !is_finite_nonnegative(settings.line_search.filter_gamma_f) ||
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

auto initialize_primal_dual_variables(const Input &input,
                                      const Settings &settings,
                                      Workspace &workspace)
    -> std::optional<double> {
  const int x_dim = input.dimensions.x_dim;
  const int s_dim = input.dimensions.s_dim;
  const int y_dim = input.dimensions.y_dim;
  double *w = workspace.csd_workspace.w;
  double *r2 = workspace.csd_workspace.r2;
  double *r3 = workspace.csd_workspace.r3;
  double *b = workspace.csd_workspace.rhs_block_3x3;
  double *v = workspace.csd_workspace.sol_block_3x3;
  double *bx = b;
  double *by = bx + x_dim;
  double *bz = by + y_dim;

  std::fill_n(w, s_dim, 1.0);
  for (int i = 0; i < y_dim; ++i) {
    r2[i] = 1.0 / workspace.penalties.y[i];
  }
  for (int i = 0; i < s_dim; ++i) {
    r3[i] = 1.0 / workspace.penalties.z[i];
  }

  std::fill_n(bx, x_dim, 0.0);
  input.add_Hx_to_y(workspace.vars.x, bx);
  const double *grad_f = input.get_grad_f();
  for (int i = 0; i < x_dim; ++i) {
    bx[i] -= grad_f[i];
  }

  std::fill_n(by, y_dim, 0.0);
  input.add_Cx_to_y(workspace.vars.x, by);
  const double *c = input.get_c();
  for (int i = 0; i < y_dim; ++i) {
    by[i] -= c[i];
  }

  std::fill_n(bz, s_dim, 0.0);
  input.add_Gx_to_y(workspace.vars.x, bz);
  const double *g = input.get_g();
  for (int i = 0; i < s_dim; ++i) {
    bz[i] -= g[i];
  }

  double initialization_psi = std::max(settings.regularization.initial,
                                       settings.regularization.first_positive);
  bool factorization_ok = false;
  for (int attempt = 0; attempt < settings.regularization.max_attempts;
       ++attempt) {
    factorization_ok = input.factor(w, initialization_psi, r2, r3, 0.0);
    if (factorization_ok) {
      break;
    }
    const double next_psi =
        increased_regularization(settings, initialization_psi);
    if (next_psi <= initialization_psi ||
        next_psi > settings.regularization.maximum) {
      break;
    }
    initialization_psi = next_psi;
  }
  if (!factorization_ok) {
    return std::nullopt;
  }

  for (int i = 0; i < x_dim; ++i) {
    bx[i] += initialization_psi * workspace.vars.x[i];
  }
  input.solve(b, v);
  std::copy_n(v, x_dim, workspace.vars.x);
  std::copy_n(v + x_dim, y_dim, workspace.vars.y);
  std::copy_n(v + x_dim + y_dim, s_dim, workspace.vars.z);

  double slack_shift = 0.0;
  double dual_shift = 0.0;
  for (int i = 0; i < s_dim; ++i) {
    workspace.vars.s[i] = -workspace.vars.z[i];
    slack_shift = std::max(slack_shift, -workspace.vars.s[i]);
    dual_shift = std::max(dual_shift, -workspace.vars.z[i]);
  }

  double mean_complementarity = 0.0;
  for (int i = 0; i < s_dim; ++i) {
    mean_complementarity += (workspace.vars.s[i] + slack_shift) *
                            (workspace.vars.z[i] + dual_shift);
  }
  if (s_dim > 0) {
    mean_complementarity = std::max(mean_complementarity / s_dim, 1e-10);
  }

  for (int i = 0; i < s_dim; ++i) {
    const double difference = workspace.vars.z[i];
    workspace.vars.z[i] =
        0.5 * (difference +
               std::sqrt(difference * difference + 4.0 * mean_complementarity));
    workspace.vars.s[i] = workspace.vars.z[i] - difference;
  }

  input.model_callback({
      .x = workspace.vars.x,
      .y = workspace.vars.y,
      .z = workspace.vars.z,
      .new_x = true,
      .new_y = true,
      .new_z = true,
  });
  return initialization_psi;
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
  constexpr double kResidualReductionFactor = 0.95;
  constexpr double kProximalResidualThreshold = 0.9;
  constexpr double kInexactProximalResidualRatio = 0.1;
  ProximalCenterUpdateRejections primal_center_update_rejections;
  ProximalCenterUpdateRejections dual_center_update_rejections;
  const double tau = settings.line_search.tau;
  const bool use_primal_center = settings.proximal.use_primal_center;
  const bool use_dual_center = settings.proximal.use_dual_center;
  const bool use_any_center = use_primal_center || use_dual_center;
  const double initial_primal_regularization_floor =
      std::max(settings.regularization.first_positive,
               settings.proximal.initial_continuation_regularization_floor);
  const double initial_dual_penalty_limit = std::min(
      settings.penalty.max_penalty_parameter,
      1.0 / settings.proximal.initial_continuation_regularization_floor);
  double primal_regularization_floor = initial_primal_regularization_floor;
  double dual_penalty_limit = initial_dual_penalty_limit;
  int num_clean_factorizations_at_continuation_floor = 0;
  double current_unregularized_kkt_residual =
      std::numeric_limits<double>::infinity();
  const bool staged_continuation_enabled =
      (use_primal_center &&
       primal_regularization_floor > settings.regularization.first_positive) ||
      (use_dual_center &&
       dual_penalty_limit < settings.penalty.max_penalty_parameter);
  bool continuation_floor_active = staged_continuation_enabled;
  bool continuation_floor_requires_stability_evidence = false;
  if (use_primal_center) {
    psi = std::max(psi, primal_regularization_floor);
  }

  if (settings.barrier.initialize_primal_dual_variables) {
    const auto initial_residuals = unregularized_residuals(input, workspace);
    std::copy_n(workspace.vars.x, x_dim, workspace.next_vars.x);
    std::copy_n(workspace.vars.s, s_dim, workspace.next_vars.s);
    std::copy_n(workspace.vars.y, y_dim, workspace.next_vars.y);
    std::copy_n(workspace.vars.z, s_dim, workspace.next_vars.z);
    const auto initialization_psi =
        initialize_primal_dual_variables(input, settings, workspace);
    if (!initialization_psi.has_value()) {
      return Output{
          .exit_status = Status::FACTORIZATION_FAILURE,
          .num_iterations = 0,
          .num_ls_iterations = 0,
          .max_primal_violation = max_primal_violation(input),
          .max_dual_violation = std::numeric_limits<double>::signaling_NaN(),
      };
    }
    add(input.get_g(), workspace.vars.s, s_dim,
        workspace.miscellaneous_workspace.g_plus_s);
    const double proximal_residual = unscaled_max_regularized_difference(
        workspace.vars.x, workspace.next_vars.x, *initialization_psi,
        input.residual_scaling.dual, x_dim);
    const bool primal_residual_dominates =
        initial_residuals.first > initial_residuals.second;
    if (primal_residual_dominates ||
        proximal_residual >
            std::max(initial_residuals.first, initial_residuals.second)) {
      std::copy_n(workspace.next_vars.x, x_dim, workspace.vars.x);
      std::copy_n(workspace.next_vars.s, s_dim, workspace.vars.s);
      std::copy_n(workspace.next_vars.y, y_dim, workspace.vars.y);
      std::copy_n(workspace.next_vars.z, s_dim, workspace.vars.z);
      input.model_callback({
          .x = workspace.vars.x,
          .y = workspace.vars.y,
          .z = workspace.vars.z,
          .new_x = true,
          .new_y = true,
          .new_z = true,
      });
      add(input.get_g(), workspace.vars.s, s_dim,
          workspace.miscellaneous_workspace.g_plus_s);
    }
  }
  if (use_primal_center) {
    std::copy_n(workspace.vars.x, x_dim, workspace.proximal_centers.x);
  }
  if (use_dual_center) {
    std::copy_n(workspace.vars.y, y_dim, workspace.proximal_centers.y);
    std::copy_n(workspace.vars.z, s_dim, workspace.proximal_centers.z);
  }
  if (settings.barrier.use_predictor_corrector) {
    mu = mean_complementarity(workspace, s_dim);
  }

  int total_ls_iterations = 0;
  // A single poorly conditioned solve can produce a spurious flat direction.
  int num_consecutive_stalled_iterations = 0;
  std::optional<double> previous_cost;
  workspace.filter.size = 0;
  bool initial_penalties_active =
      settings.penalty.initialize_from_linearized_constraint_reduction;

  for (int iteration = 0; iteration < settings.max_iterations; ++iteration) {
    bool regularization_at_continuation_floor = false;
    if (continuation_floor_active) {
      const double dual_proximal_residual =
          use_primal_center
              ? unscaled_max_regularized_difference(
                    workspace.vars.x, workspace.proximal_centers.x, psi,
                    input.residual_scaling.dual, x_dim)
              : 0.0;
      const double primal_proximal_residual =
          use_dual_center
              ? std::max(unscaled_max_inverse_regularized_difference(
                             workspace.vars.y, workspace.proximal_centers.y,
                             workspace.penalties.y,
                             input.residual_scaling.equality, y_dim),
                         unscaled_max_inverse_regularized_difference(
                             workspace.vars.z, workspace.proximal_centers.z,
                             workspace.penalties.z,
                             input.residual_scaling.inequality, s_dim))
              : 0.0;
      bool dual_regularization_at_limit = true;
      for (int i = 0; i < y_dim; ++i) {
        dual_regularization_at_limit &=
            workspace.penalties.y[i] == dual_penalty_limit;
      }
      for (int i = 0; i < s_dim; ++i) {
        dual_regularization_at_limit &=
            workspace.penalties.z[i] == dual_penalty_limit;
      }
      regularization_at_continuation_floor =
          (!use_primal_center || psi == primal_regularization_floor) &&
          (!use_dual_center || dual_regularization_at_limit);
      if (regularization_at_continuation_floor &&
          (!continuation_floor_requires_stability_evidence ||
           num_clean_factorizations_at_continuation_floor > 0 ||
           current_unregularized_kkt_residual < kProximalResidualThreshold) &&
          dual_proximal_residual < kProximalResidualThreshold &&
          primal_proximal_residual < kProximalResidualThreshold) {
        primal_regularization_floor = settings.regularization.first_positive;
        dual_penalty_limit = settings.penalty.max_penalty_parameter;
        continuation_floor_active = false;
        continuation_floor_requires_stability_evidence = false;
      }
    }

    const double f0 = input.get_f();

    const double ctc = squared_norm(input.get_c(), y_dim);
    const double gsetgse =
        squared_norm(workspace.miscellaneous_workspace.g_plus_s, s_dim);

    const double sq_constraint_violation_norm = ctc + gsetgse;

    if (iteration > 0 && initial_penalties_active) {
      const double max_violation =
          settings.termination.max_constraint_violation;
      const bool primal_feasibility_satisfied =
          unscaled_max_abs(input.get_c(), input.residual_scaling.equality,
                           y_dim) <= max_violation &&
          unscaled_max_abs(workspace.miscellaneous_workspace.g_plus_s,
                           input.residual_scaling.inequality,
                           s_dim) <= max_violation;
      if (primal_feasibility_satisfied) {
        std::fill_n(workspace.penalties.y, y_dim,
                    settings.penalty.initial_penalty_parameter);
        std::fill_n(workspace.penalties.z, s_dim,
                    settings.penalty.initial_penalty_parameter);
        initial_penalties_active = false;
      }
    }

    auto search_direction =
        compute_search_direction(input, settings, mu, psi, tau, workspace);
    double previous_linearized_violation_ratio =
        std::numeric_limits<double>::infinity();
    while (
        iteration == 0 && std::get<0>(search_direction) &&
        settings.penalty.initialize_from_linearized_constraint_reduction &&
        update_initial_penalties_for_linearized_constraint_violation(
            input, settings, previous_linearized_violation_ratio, workspace)) {
      search_direction =
          compute_search_direction(input, settings, mu, psi, tau, workspace);
    }
    const auto [factorization_ok, dx, ds, dy, dz, stationarity_merit_slope,
                merit_slope, alpha_s_max, dual_residual,
                max_constraint_violation, max_complementarity, kkt_error,
                duality_gap, lin_sys_error, num_regularization_increases] =
        search_direction;

    if (!factorization_ok) {
      return Output{
          .exit_status = Status::FACTORIZATION_FAILURE,
          .num_iterations = iteration,
          .num_ls_iterations = total_ls_iterations,
          .max_primal_violation = max_primal_violation(input),
          .max_dual_violation = std::numeric_limits<double>::signaling_NaN(),
      };
    }
    if (continuation_floor_active && regularization_at_continuation_floor &&
        num_regularization_increases == 0) {
      ++num_clean_factorizations_at_continuation_floor;
    } else {
      num_clean_factorizations_at_continuation_floor = 0;
    }
    const bool restore_continuation_floor =
        staged_continuation_enabled && num_regularization_increases > 0 &&
        kkt_error >= kProximalResidualThreshold;

    const double alpha_z_max = get_fraction_to_boundary_step(
        s_dim, tau, workspace.vars.z, workspace.delta_vars.z);

    const TerminationChecks termination = check_termination(
        settings, previous_cost, f0, stationarity_merit_slope, dual_residual,
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

    if (termination.stalled && !settings.barrier.use_predictor_corrector &&
        !use_any_center) {
      ++num_consecutive_stalled_iterations;
    } else {
      num_consecutive_stalled_iterations = 0;
    }

    if (termination.solved ||
        num_consecutive_stalled_iterations >=
            settings.termination
                .num_consecutive_stalled_iterations_before_termination) {
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

    if (termination.advance_barrier &&
        !settings.barrier.use_predictor_corrector) {
      const double next_mu = std::max(mu * settings.barrier.mu_update_factor,
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
      const double dual_step =
          settings.barrier.use_predictor_corrector ? alpha_z_max : alpha;
      update_next_dual_vars(input, settings, tau, workspace, dual_step);
      ls_succeeded = true;
      m0 = 0.0;
    } else {
      std::tie(ls_succeeded, alpha, m0, std::ignore) = do_line_search(
          input, settings, mu, psi, tau, sq_constraint_violation_norm,
          merit_slope, alpha_s_max, alpha_z_max, total_ls_iterations,
          workspace.filter, workspace);
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

    const double previous_complementarity =
        mean_complementarity(workspace, s_dim);
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

    const double new_complementarity = mean_complementarity(workspace, s_dim);
    double new_primal_residual = 0.0;
    double new_dual_residual = 0.0;
    if (use_any_center) {
      std::tie(new_primal_residual, new_dual_residual) =
          unregularized_residuals(input, workspace);
      current_unregularized_kkt_residual = std::max(
          {new_primal_residual, new_dual_residual, new_complementarity});
    }
    const auto relative_reduction = [](const double previous,
                                       const double current) {
      return previous > 0.0
                 ? std::clamp((previous - current) / previous, 0.0, 1.0)
                 : 0.0;
    };
    const double proximal_regularization_reduction =
        relative_reduction(std::max({dual_residual, max_constraint_violation,
                                     previous_complementarity}),
                           std::max({new_dual_residual, new_primal_residual,
                                     new_complementarity}));

    if (use_primal_center) {
      for (int i = 0; i < x_dim; ++i) {
        workspace.nrhs.x[i] +=
            psi * (workspace.vars.x[i] - workspace.proximal_centers.x[i]);
      }
      const double regularized_dual_residual = unscaled_max_abs(
          workspace.nrhs.x, input.residual_scaling.dual, x_dim);
      const double new_dual_proximal_residual =
          unscaled_max_regularized_difference(
              workspace.vars.x, workspace.proximal_centers.x, psi,
              input.residual_scaling.dual, x_dim);

      if (!settings.proximal.update_centers) {
        psi = std::max(primal_regularization_floor,
                       settings.regularization.decrease_factor * psi);
      } else if (new_dual_residual < kResidualReductionFactor * dual_residual ||
                 new_dual_residual < settings.termination.max_dual_residual ||
                 (psi == primal_regularization_floor &&
                  new_dual_proximal_residual < kProximalResidualThreshold)) {
        std::copy_n(workspace.vars.x, x_dim, workspace.proximal_centers.x);
        primal_center_update_rejections.reset();
        psi = std::max(primal_regularization_floor,
                       (1.0 - proximal_regularization_reduction) * psi);
      } else {
        primal_center_update_rejections.reject(dual_residual);
        if (iteration < 5 ||
            new_dual_proximal_residual < kProximalResidualThreshold) {
          psi =
              std::max(primal_regularization_floor,
                       (1.0 - 0.666 * proximal_regularization_reduction) * psi);
        }
        if (primal_center_update_rejections.has_reduced_residual(
                new_dual_residual, kResidualReductionFactor)) {
          primal_center_update_rejections.reset();
        } else if (primal_center_update_rejections.count >=
                       settings.proximal
                           .max_consecutive_center_update_rejections &&
                   regularized_dual_residual <
                       std::max(settings.termination.max_dual_residual,
                                kInexactProximalResidualRatio *
                                    new_dual_residual)) {
          std::copy_n(workspace.vars.x, x_dim, workspace.proximal_centers.x);
          primal_center_update_rejections.reset();
        }
      }
    } else {
      psi = decreased_regularization(settings, psi);
    }

    if (use_dual_center) {
      const double *new_c = input.get_c();
      double *r2 = workspace.csd_workspace.r2;
      double *r3 = workspace.csd_workspace.r3;
      for (int i = 0; i < y_dim; ++i) {
        workspace.nrhs.y[i] =
            new_c[i] +
            r2[i] * (workspace.proximal_centers.y[i] - workspace.vars.y[i]);
      }
      for (int i = 0; i < s_dim; ++i) {
        workspace.nrhs.z[i] =
            workspace.miscellaneous_workspace.g_plus_s[i] +
            r3[i] * (workspace.proximal_centers.z[i] - workspace.vars.z[i]);
      }
      const double regularized_primal_residual =
          std::max(unscaled_max_abs(workspace.nrhs.y,
                                    input.residual_scaling.equality, y_dim),
                   unscaled_max_abs(workspace.nrhs.z,
                                    input.residual_scaling.inequality, s_dim));
      const double new_primal_proximal_residual =
          std::max(unscaled_max_regularized_difference(
                       workspace.vars.y, workspace.proximal_centers.y, r2,
                       input.residual_scaling.equality, y_dim),
                   unscaled_max_regularized_difference(
                       workspace.vars.z, workspace.proximal_centers.z, r3,
                       input.residual_scaling.inequality, s_dim));
      bool dual_regularization_at_minimum = true;
      for (int i = 0; i < y_dim; ++i) {
        dual_regularization_at_minimum &=
            workspace.penalties.y[i] == dual_penalty_limit;
      }
      for (int i = 0; i < s_dim; ++i) {
        dual_regularization_at_minimum &=
            workspace.penalties.z[i] == dual_penalty_limit;
      }

      if (!settings.proximal.update_centers) {
        decrease_dual_regularization(
            input, workspace, dual_penalty_limit,
            1.0 / settings.penalty.penalty_parameter_increase_factor);
      } else if (new_primal_residual <
                     kResidualReductionFactor * max_constraint_violation ||
                 new_primal_residual <
                     settings.termination.max_constraint_violation ||
                 (dual_regularization_at_minimum &&
                  new_primal_proximal_residual < kProximalResidualThreshold)) {
        std::copy_n(workspace.vars.y, y_dim, workspace.proximal_centers.y);
        std::copy_n(workspace.vars.z, s_dim, workspace.proximal_centers.z);
        dual_center_update_rejections.reset();
        decrease_dual_regularization(input, workspace, dual_penalty_limit,
                                     1.0 - proximal_regularization_reduction);
      } else {
        dual_center_update_rejections.reject(max_constraint_violation);
        if (iteration < 5 ||
            new_primal_proximal_residual < kProximalResidualThreshold) {
          decrease_dual_regularization(
              input, workspace, dual_penalty_limit,
              1.0 - 0.666 * proximal_regularization_reduction);
        }
        if (dual_center_update_rejections.has_reduced_residual(
                new_primal_residual, kResidualReductionFactor)) {
          dual_center_update_rejections.reset();
        } else if (dual_center_update_rejections.count >=
                       settings.proximal
                           .max_consecutive_center_update_rejections &&
                   regularized_primal_residual <
                       std::max(settings.termination.max_constraint_violation,
                                kInexactProximalResidualRatio *
                                    new_primal_residual)) {
          std::copy_n(workspace.vars.y, y_dim, workspace.proximal_centers.y);
          std::copy_n(workspace.vars.z, s_dim, workspace.proximal_centers.z);
          dual_center_update_rejections.reset();
        }
      }
    } else {
      const bool any_penalty_increased =
          update_penalty_parameters(input, settings, workspace);
      if (any_penalty_increased) {
        // Reset regularization when penalty increases, to stabilize the
        // modified KKT system.
        psi = std::max(psi, settings.regularization.initial);
      }
    }

    if (restore_continuation_floor) {
      primal_regularization_floor = initial_primal_regularization_floor;
      dual_penalty_limit = initial_dual_penalty_limit;
      continuation_floor_active = true;
      continuation_floor_requires_stability_evidence = true;
      num_clean_factorizations_at_continuation_floor = 0;
      if (use_primal_center) {
        psi = std::max(psi, primal_regularization_floor);
      }
      if (use_dual_center) {
        for (int i = 0; i < y_dim; ++i) {
          workspace.penalties.y[i] =
              std::min(workspace.penalties.y[i], dual_penalty_limit);
        }
        for (int i = 0; i < s_dim; ++i) {
          workspace.penalties.z[i] =
              std::min(workspace.penalties.z[i], dual_penalty_limit);
        }
      }
    }

    if (settings.barrier.use_predictor_corrector) {
      mu = new_complementarity;
    } else if (kkt_error <= settings.barrier.mu_update_kappa * mu) {
      mu = std::max(mu * settings.barrier.mu_update_factor,
                    settings.barrier.mu_min);
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
