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

constexpr auto uses_primal_center(const Mode mode) -> bool {
  return mode != Mode::REGULARIZED_IPM;
}

constexpr auto uses_dual_center(const Mode mode) -> bool {
  return mode == Mode::PRIMAL_DUAL_PROXIMAL_IPM;
}

auto proximal_centers(const Settings &settings, const Workspace &workspace)
    -> const VariablesWorkspace & {
  return settings.barrier.use_predictor_corrector ? workspace.proximal_centers
                                                  : workspace.vars;
}

template <typename Callback>
void for_each_bound_side(const Input &input, const int *bound_sides,
                         const int num_bound_sides, Callback &&callback) {
  for (int side_index = 0; side_index < num_bound_sides; ++side_index) {
    const int encoded_side = bound_sides[side_index];
    const int variable_index = encoded_side / 2;
    const bool is_upper = encoded_side % 2 != 0;
    const double endpoint = is_upper ? input.upper_bounds[variable_index]
                                     : input.lower_bounds[variable_index];
    const double jacobian = is_upper ? 1.0 : -1.0;
    callback(side_index, variable_index, endpoint, jacobian);
  }
}

auto initialize_bound_sides(const Input &input, int *bound_sides) -> int {
  if (input.lower_bounds == nullptr && input.upper_bounds == nullptr) {
    return 0;
  }
  int side_index = 0;
  for (int variable_index = 0; variable_index < input.dimensions.x_dim;
       ++variable_index) {
    if (input.lower_bounds != nullptr &&
        std::isfinite(input.lower_bounds[variable_index])) {
      bound_sides[side_index] = 2 * variable_index;
      ++side_index;
    }
    if (input.upper_bounds != nullptr &&
        std::isfinite(input.upper_bounds[variable_index])) {
      bound_sides[side_index] = 2 * variable_index + 1;
      ++side_index;
    }
  }
  return side_index;
}

auto bound_constraint_value(const double x, const double endpoint,
                            const double jacobian) -> double {
  return jacobian * (x - endpoint);
}

void set_bound_residuals(const Input &input, const double *x,
                         const double *bound_s, const int *bound_sides,
                         const int num_bound_sides, double *bound_g_plus_s) {
  for_each_bound_side(input, bound_sides, num_bound_sides,
                      [&](const int side_index, const int variable_index,
                          const double endpoint, const double jacobian) {
                        bound_g_plus_s[side_index] =
                            bound_constraint_value(x[variable_index], endpoint,
                                                   jacobian) +
                            bound_s[side_index];
                      });
}

auto mean_penalty_parameter(const Workspace &workspace, const int s_dim,
                            const int y_dim, const int num_bound_sides)
    -> double {
  double sum = 0.0;
  for (int i = 0; i < y_dim; ++i) {
    sum += workspace.penalties.y[i];
  }
  for (int i = 0; i < s_dim; ++i) {
    sum += workspace.penalties.z[i];
  }
  for (int i = 0; i < num_bound_sides; ++i) {
    sum += workspace.penalties.bound_z[i];
  }
  const int dim = s_dim + y_dim + num_bound_sides;
  return dim == 0 ? 0.0 : sum / dim;
}

auto mean_complementarity(const Workspace &workspace, const int s_dim,
                          const int num_bound_sides) -> double {
  const int dim = s_dim + num_bound_sides;
  if (dim == 0) {
    return 0.0;
  }
  return (dot(workspace.vars.s, workspace.vars.z, s_dim) +
          dot(workspace.vars.bound_s, workspace.vars.bound_z,
              num_bound_sides)) /
         dim;
}

auto max_primal_violation(const Input &input, const Workspace &workspace,
                          const int num_bound_sides, const double *x)
    -> double {
  const int s_dim = input.dimensions.s_dim;
  const int y_dim = input.dimensions.y_dim;
  double result = std::max(max_abs_or_inf(input.get_c(), y_dim),
                           max_positive_or_inf(input.get_g(), s_dim));
  for_each_bound_side(input, workspace.bound_sides, num_bound_sides,
                      [&](const int, const int variable_index,
                          const double endpoint, const double jacobian) {
                        const double violation = bound_constraint_value(
                            x[variable_index], endpoint, jacobian);
                        result = std::isnan(violation)
                                     ? std::numeric_limits<double>::infinity()
                                     : std::max(result, violation);
                      });
  return result;
}

auto unregularized_residuals(const Input &input, Workspace &workspace)
    -> std::pair<double, double> {
  const int x_dim = input.dimensions.x_dim;
  const int s_dim = input.dimensions.s_dim;
  const int y_dim = input.dimensions.y_dim;
  const int num_bound_sides = workspace.num_bound_sides;

  double *dual = workspace.nrhs.x;
  std::copy_n(input.get_grad_f(), x_dim, dual);
  input.add_CTx_to_y(workspace.vars.y, dual);
  input.add_GTx_to_y(workspace.vars.z, dual);
  for_each_bound_side(input, workspace.bound_sides, num_bound_sides,
                      [&](const int side_index, const int variable_index,
                          const double, const double jacobian) {
                        dual[variable_index] +=
                            jacobian * workspace.vars.bound_z[side_index];
                      });

  const double primal = std::max(
      {max_abs_or_inf(input.get_c(), y_dim),
       max_abs_or_inf(workspace.miscellaneous_workspace.g_plus_s, s_dim),
       max_abs_or_inf(workspace.miscellaneous_workspace.bound_g_plus_s,
                      num_bound_sides)});
  return {primal, max_abs_or_inf(dual, x_dim)};
}

auto merit_function(const Input &input, const Settings &settings,
                    const Workspace &workspace, const double *x,
                    const double *s, const double *y, const double *z,
                    const double *bound_s, const double *bound_z,
                    const double *g_plus_s, const double *bound_g_plus_s,
                    const double mu, const double psi)
    -> std::tuple<double, double, double, double, double, double> {
  const int x_dim = input.dimensions.x_dim;
  const int s_dim = input.dimensions.s_dim;
  const int y_dim = input.dimensions.y_dim;
  const int num_bound_sides = workspace.num_bound_sides;
  const double *c = input.get_c();
  const VariablesWorkspace &centers = proximal_centers(settings, workspace);
  double s_term = -mu * sum_of_logs(s, s_dim);
  double c_term;
  double g_term;
  double aug_term;

  if (uses_dual_center(settings.mode)) {
    // The PDAL merit adds both the constraint and dual-proximal residual
    // squared norms, weighted by the augmented-Lagrangian penalties.
    c_term = 0.0;
    g_term = 0.0;
    aug_term = 0.0;
    for (int i = 0; i < y_dim; ++i) {
      const double eta = workspace.penalties.y[i];
      const double regularized_residual = c[i] - (y[i] - centers.y[i]) / eta;
      c_term += centers.y[i] * c[i];
      aug_term += 0.5 * eta *
                  (c[i] * c[i] + regularized_residual * regularized_residual);
    }
    for (int i = 0; i < s_dim; ++i) {
      const double eta = workspace.penalties.z[i];
      const double regularized_residual =
          g_plus_s[i] - (z[i] - centers.z[i]) / eta;
      g_term += centers.z[i] * g_plus_s[i];
      aug_term += 0.5 * eta *
                  (g_plus_s[i] * g_plus_s[i] +
                   regularized_residual * regularized_residual);
    }
    for (int i = 0; i < num_bound_sides; ++i) {
      const double eta = workspace.penalties.bound_z[i];
      const double regularized_residual =
          bound_g_plus_s[i] - (bound_z[i] - centers.bound_z[i]) / eta;
      g_term += centers.bound_z[i] * bound_g_plus_s[i];
      aug_term += 0.5 * eta *
                  (bound_g_plus_s[i] * bound_g_plus_s[i] +
                   regularized_residual * regularized_residual);
    }
  } else {
    c_term = dot(c, y, y_dim);
    g_term = dot(g_plus_s, z, s_dim);
    aug_term =
        0.5 * (weighted_squared_norm(c, workspace.penalties.y, y_dim) +
               weighted_squared_norm(g_plus_s, workspace.penalties.z, s_dim));
    g_term += dot(bound_g_plus_s, bound_z, num_bound_sides);
    aug_term +=
        0.5 * weighted_squared_norm(bound_g_plus_s, workspace.penalties.bound_z,
                                    num_bound_sides);
  }

  if (uses_primal_center(settings.mode)) {
    for (int i = 0; i < x_dim; ++i) {
      const double displacement = x[i] - centers.x[i];
      aug_term += 0.5 * psi * displacement * displacement;
    }
  }

  s_term -= mu * sum_of_logs(bound_s, num_bound_sides);

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
             "{:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10}\n",
             // clang-format on
             "", "linsys_res", "alpha_s_m", "alpha_z_m", "m_slope", "obs_slope",
             "m_sl[x]", "obs_sl[x]", "m_sl[s]", "obs_sl[s]", "m_sl[y]",
             "obs_sl[y]", "m_sl[z]", "obs_sl[z]", "|nrhs_x|", "|nrhs_s|");
}

void print_line_search_log_header() {
  fmt::print(fmt::emphasis::bold | fg(fmt::color::yellow),
             // clang-format off
             "{:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10}\n",
             // clang-format on
             "", "ls_iteration", "alpha", "merit", "f", "|c|", "|g+s|", "dm",
             "dm/alpha", "dm[x]", "dm[s]", "dm[y]", "dm[z]");
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
  return uses_primal_center(settings.mode) ? regularization.first_positive
                                           : 0.0;
}

struct TerminationChecks {
  bool solved;
  bool stalled;
  bool advance_barrier;
};

struct CenterUpdateRejections {
  void reject(const double residual) {
    if (count == 0) {
      residual_at_start = residual;
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
      !settings.barrier.use_predictor_corrector &&
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
  const int num_bound_sides = workspace.num_bound_sides;
  bool any_increased = false;

  const double *old_c = workspace.nrhs.y;
  const double *old_gps = workspace.nrhs.z;
  const double *old_bound_gps = workspace.nrhs.bound_z;
  const double *new_c = input.get_c();
  const double *new_gps = workspace.miscellaneous_workspace.g_plus_s;
  const double *new_bound_gps =
      workspace.miscellaneous_workspace.bound_g_plus_s;
  const double min_ratio =
      settings.penalty.min_acceptable_constraint_violation_ratio;

  const auto update = [&](const double *old_residual,
                          const double *new_residual, double *penalty,
                          const int size) {
    for (int i = 0; i < size; ++i) {
      const double improvement_ratio =
          new_residual[i] * new_residual[i] /
          std::max(old_residual[i] * old_residual[i], 1e-12);
      const double factor =
          improvement_ratio > min_ratio
              ? settings.penalty.penalty_parameter_increase_factor
              : settings.penalty.penalty_parameter_decrease_factor;
      penalty[i] =
          std::min(penalty[i] * factor, settings.penalty.max_penalty_parameter);
      any_increased |= improvement_ratio > min_ratio;
    }
  };
  update(old_c, new_c, workspace.penalties.y, y_dim);
  update(old_gps, new_gps, workspace.penalties.z, s_dim);
  update(old_bound_gps, new_bound_gps, workspace.penalties.bound_z,
         num_bound_sides);

  return any_increased;
}

void update_next_primal_vars(const Input &input, const double tau,
                             Workspace &workspace, const double alpha,
                             const bool update_x, const bool update_s) {
  const int x_dim = input.dimensions.x_dim;
  const int s_dim = input.dimensions.s_dim;
  const int num_bound_sides = workspace.num_bound_sides;

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
    for (int i = 0; i < num_bound_sides; ++i) {
      workspace.next_vars.bound_s[i] = std::max(
          workspace.vars.bound_s[i] + alpha * workspace.delta_vars.bound_s[i],
          (1.0 - tau) * workspace.vars.bound_s[i]);
    }
  } else {
    std::copy_n(workspace.vars.s, s_dim, workspace.next_vars.s);
    if (num_bound_sides > 0) {
      std::copy_n(workspace.vars.bound_s, num_bound_sides,
                  workspace.next_vars.bound_s);
    }
  }

  add(input.get_g(), workspace.next_vars.s, s_dim,
      workspace.miscellaneous_workspace.g_plus_s);
  set_bound_residuals(input, workspace.next_vars.x, workspace.next_vars.bound_s,
                      workspace.bound_sides, num_bound_sides,
                      workspace.miscellaneous_workspace.bound_g_plus_s);
}

void update_next_dual_vars(const Input &input, const double tau,
                           Workspace &workspace, const double alpha,
                           const bool update_y, const bool update_z) {
  const int s_dim = input.dimensions.s_dim;
  const int y_dim = input.dimensions.y_dim;
  const int num_bound_sides = workspace.num_bound_sides;

  if (update_y) {
    for (int i = 0; i < y_dim; ++i) {
      workspace.next_vars.y[i] =
          workspace.vars.y[i] + alpha * workspace.delta_vars.y[i];
    }
  } else {
    std::copy_n(workspace.vars.y, y_dim, workspace.next_vars.y);
  }

  if (update_z) {
    for (int i = 0; i < s_dim; ++i) {
      const double next_z =
          workspace.vars.z[i] + alpha * workspace.delta_vars.z[i];
      workspace.next_vars.z[i] =
          std::max(next_z, (1.0 - tau) * workspace.vars.z[i]);
    }
    for (int i = 0; i < num_bound_sides; ++i) {
      const double next_bound_z =
          workspace.vars.bound_z[i] + alpha * workspace.delta_vars.bound_z[i];
      workspace.next_vars.bound_z[i] =
          std::max(next_bound_z, (1.0 - tau) * workspace.vars.bound_z[i]);
    }
  } else {
    std::copy_n(workspace.vars.z, s_dim, workspace.next_vars.z);
    if (num_bound_sides > 0) {
      std::copy_n(workspace.vars.bound_z, num_bound_sides,
                  workspace.next_vars.bound_z);
    }
  }

  ModelCallbackInput mci{
      .x = workspace.next_vars.x,
      .y = workspace.next_vars.y,
      .z = workspace.next_vars.z,
      .new_x = false,
      .new_y = update_y,
      .new_z = update_z,
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
    delta_vars_tmp.reserve(x_dim, s_dim, y_dim, workspace.num_bound_sides);
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

struct MeritSlope {
  double x;
  double s;
  double y;
  double z;
  double total;
};

auto get_observed_merit_slope(const Input &input, const Settings &settings,
                              const double mu, const double psi,
                              const double tau, Workspace &workspace)
    -> MeritSlope {
  const auto compute_empirical_merit_slope = [&](const bool update_x,
                                                 const bool update_s,
                                                 const bool update_y,
                                                 const bool update_z) {
    const auto get_perturbed_merit = [&](const double beta) -> double {
      update_next_primal_vars(input, tau, workspace, beta, update_x, update_s);
      if (update_y || update_z) {
        update_next_dual_vars(input, tau, workspace, beta, update_y, update_z);
      }
      const double *y = update_y ? workspace.next_vars.y : workspace.vars.y;
      const double *z = update_z ? workspace.next_vars.z : workspace.vars.z;
      const double *bound_z =
          update_z ? workspace.next_vars.bound_z : workspace.vars.bound_z;
      const auto [_mP_f, _mP_s, _mP_c, _mP_g, _mP_aug, mP] = merit_function(
          input, settings, workspace, workspace.next_vars.x,
          workspace.next_vars.s, y, z, workspace.next_vars.bound_s, bound_z,
          workspace.miscellaneous_workspace.g_plus_s,
          workspace.miscellaneous_workspace.bound_g_plus_s, mu, psi);
      return mP;
    };

    const double h = std::sqrt(std::numeric_limits<double>::epsilon());

    const double mP = get_perturbed_merit(h);
    const double mM = get_perturbed_merit(-h);

    return (mP - mM) / (2 * h);
  };

  const bool update_dual = uses_dual_center(settings.mode);
  const double os_x = compute_empirical_merit_slope(true, false, false, false);
  const double os_s = compute_empirical_merit_slope(false, true, false, false);
  const double os_y =
      update_dual ? compute_empirical_merit_slope(false, false, true, false)
                  : 0.0;
  const double os_z =
      update_dual ? compute_empirical_merit_slope(false, false, false, true)
                  : 0.0;
  const double os =
      compute_empirical_merit_slope(true, true, update_dual, update_dual);

  update_next_primal_vars(input, tau, workspace, 0.0, false, false);
  if (update_dual) {
    update_next_dual_vars(input, tau, workspace, 0.0, true, true);
  }

  return {.x = os_x, .s = os_s, .y = os_y, .z = os_z, .total = os};
}

auto merit_slope(const Input &input, const Settings &settings,
                 const Workspace &workspace, const double mu, const double psi,
                 const double *dx, const double *ds, const double *dy,
                 const double *dz) -> MeritSlope {
  const int x_dim = input.dimensions.x_dim;
  const int s_dim = input.dimensions.s_dim;
  const int y_dim = input.dimensions.y_dim;
  const int num_bound_sides = workspace.num_bound_sides;
  const double *c = input.get_c();
  const double *gps = workspace.miscellaneous_workspace.g_plus_s;
  const double *bound_gps = workspace.miscellaneous_workspace.bound_g_plus_s;
  const double *bound_ds = workspace.delta_vars.bound_s;
  const double *bound_dz = workspace.delta_vars.bound_z;
  const VariablesWorkspace &centers = proximal_centers(settings, workspace);

  if (uses_dual_center(settings.mode)) {
    double *gradient_x = workspace.next_vars.x;
    double *multiplier_y = workspace.next_vars.y;
    double *multiplier_z = workspace.next_vars.z;
    double *multiplier_bound_z = workspace.next_vars.bound_z;

    std::copy_n(input.get_grad_f(), x_dim, gradient_x);
    for (int i = 0; i < x_dim; ++i) {
      gradient_x[i] += psi * (workspace.vars.x[i] - centers.x[i]);
    }
    double y_slope = 0.0;
    for (int i = 0; i < y_dim; ++i) {
      const double eta = workspace.penalties.y[i];
      multiplier_y[i] = centers.y[i] + 2.0 * eta * c[i];
      y_slope -= c[i] * dy[i];
    }
    double z_slope = 0.0;
    for (int i = 0; i < s_dim; ++i) {
      const double eta = workspace.penalties.z[i];
      multiplier_z[i] = centers.z[i] + 2.0 * eta * gps[i];
      z_slope -= gps[i] * dz[i];
    }
    for (int i = 0; i < num_bound_sides; ++i) {
      const double eta = workspace.penalties.bound_z[i];
      multiplier_bound_z[i] = centers.bound_z[i] + 2.0 * eta * bound_gps[i];
      z_slope -= bound_gps[i] * bound_dz[i];
    }
    input.add_CTx_to_y(multiplier_y, gradient_x);
    input.add_GTx_to_y(multiplier_z, gradient_x);
    for_each_bound_side(input, workspace.bound_sides, num_bound_sides,
                        [&](const int side_index, const int variable_index,
                            const double, const double jacobian) {
                          gradient_x[variable_index] +=
                              jacobian * multiplier_bound_z[side_index];
                        });

    const double x_slope = dot(gradient_x, dx, x_dim);
    double s_slope = 0.0;
    for (int i = 0; i < s_dim; ++i) {
      s_slope += (multiplier_z[i] - mu / workspace.vars.s[i]) * ds[i];
    }
    for (int i = 0; i < num_bound_sides; ++i) {
      s_slope += (multiplier_bound_z[i] - mu / workspace.vars.bound_s[i]) *
                 bound_ds[i];
    }
    return {.x = x_slope,
            .s = s_slope,
            .y = y_slope,
            .z = z_slope,
            .total = x_slope + s_slope + y_slope + z_slope};
  }

  double *tmp_y = workspace.next_vars.y;
  double *tmp_s = workspace.next_vars.s;
  double *tmp_bound_s = workspace.next_vars.bound_s;

  std::fill_n(tmp_y, y_dim, 0.0);
  input.add_Cx_to_y(dx, tmp_y);

  std::fill_n(tmp_s, s_dim, 0.0);
  input.add_Gx_to_y(dx, tmp_s);

  for_each_bound_side(input, workspace.bound_sides, num_bound_sides,
                      [&](const int side_index, const int variable_index,
                          const double, const double jacobian) {
                        tmp_bound_s[side_index] = jacobian * dx[variable_index];
                      });

  double x_slope = dot(workspace.nrhs.x, dx, x_dim) +
                   weighted_dot(c, workspace.penalties.y, tmp_y, y_dim) +
                   weighted_dot(gps, workspace.penalties.z, tmp_s, s_dim) +
                   weighted_dot(bound_gps, workspace.penalties.bound_z,
                                tmp_bound_s, num_bound_sides);
  if (uses_primal_center(settings.mode)) {
    for (int i = 0; i < x_dim; ++i) {
      x_slope += psi * (workspace.vars.x[i] - centers.x[i]) * dx[i];
    }
  }
  const double s_slope =
      dot(workspace.nrhs.s, ds, s_dim) +
      weighted_dot(gps, workspace.penalties.z, ds, s_dim) +
      dot(workspace.nrhs.bound_s, bound_ds, num_bound_sides) +
      weighted_dot(bound_gps, workspace.penalties.bound_z, bound_ds,
                   num_bound_sides);

  return {.x = x_slope,
          .s = s_slope,
          .y = 0.0,
          .z = 0.0,
          .total = x_slope + s_slope};
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
  const int num_bound_sides = workspace.num_bound_sides;
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
  const double *bound_s = workspace.vars.bound_s;
  const double *bound_z = workspace.vars.bound_z;
  const VariablesWorkspace &centers = proximal_centers(settings, workspace);

  const double *grad_f = input.get_grad_f();

  const double *c = input.get_c();
  const double *gps = workspace.miscellaneous_workspace.g_plus_s;
  const double *bound_gps = workspace.miscellaneous_workspace.bound_g_plus_s;

  double *dx = workspace.delta_vars.x;
  double *ds = workspace.delta_vars.s;
  double *dy = workspace.delta_vars.y;
  double *dz = workspace.delta_vars.z;
  double *bound_ds = workspace.delta_vars.bound_s;
  double *bound_dz = workspace.delta_vars.bound_z;

  double *rx = workspace.nrhs.x;
  double *rs = workspace.nrhs.s;
  double *ry = workspace.nrhs.y;
  double *rz = workspace.nrhs.z;
  double *bound_rs = workspace.nrhs.bound_s;
  double *bound_rz = workspace.nrhs.bound_z;

  double *r1 = workspace.csd_workspace.r1;
  double *w = workspace.csd_workspace.w;
  double *r2 = workspace.csd_workspace.r2;
  double *r3 = workspace.csd_workspace.r3;
  double *bound_w = workspace.csd_workspace.bound_w;
  double *bound_r3 = workspace.csd_workspace.bound_r3;
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

  const bool use_exact_barrier_diagonal =
      settings.barrier.use_predictor_corrector &&
      uses_primal_center(settings.mode);
  for (int i = 0; i < s_dim; ++i) {
    const double ratio = s[i] / z[i];
    w[i] = use_exact_barrier_diagonal
               ? std::clamp(ratio, std::numeric_limits<double>::min(),
                            std::numeric_limits<double>::max())
               : std::clamp(ratio, 1e-18, 1e18);
  }

  for (int i = 0; i < y_dim; ++i) {
    r2[i] = 1.0 / workspace.penalties.y[i];
  }
  for (int i = 0; i < s_dim; ++i) {
    r3[i] = 1.0 / workspace.penalties.z[i];
  }
  std::fill_n(r1, x_dim, psi);
  for_each_bound_side(
      input, workspace.bound_sides, num_bound_sides,
      [&](const int side_index, const int variable_index, const double,
          const double) {
        const double ratio = bound_s[side_index] / bound_z[side_index];
        bound_w[side_index] =
            use_exact_barrier_diagonal
                ? std::clamp(ratio, std::numeric_limits<double>::min(),
                             std::numeric_limits<double>::max())
                : std::clamp(ratio, 1e-18, 1e18);
        bound_r3[side_index] = 1.0 / workspace.penalties.bound_z[side_index];
        bound_rz[side_index] =
            bound_gps[side_index] +
            (uses_dual_center(settings.mode)
                 ? bound_r3[side_index] *
                       (centers.bound_z[side_index] - bound_z[side_index])
                 : 0.0);
        r1[variable_index] +=
            1.0 / (bound_w[side_index] + bound_r3[side_index]);
      });

  int num_regularization_increases = 0;
  bool factorization_ok = false;
  for (int attempt = 0; attempt < settings.regularization.max_attempts;
       ++attempt) {
    factorization_ok = input.factor(w, r1, r2, r3);
    if (factorization_ok) {
      break;
    }
    const double next_psi = increased_regularization(settings, psi);
    if (next_psi <= psi || next_psi > settings.regularization.maximum) {
      break;
    }
    const double increase = next_psi - psi;
    for (int i = 0; i < x_dim; ++i) {
      r1[i] += increase;
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
  for_each_bound_side(input, workspace.bound_sides, num_bound_sides,
                      [&](const int side_index, const int variable_index,
                          const double, const double jacobian) {
                        rx[variable_index] += jacobian * bound_z[side_index];
                      });

  if (std::isfinite(settings.termination.max_duality_gap)) {
    const double *g = input.get_g();
    duality_gap =
        dot(workspace.vars.x, rx, x_dim) - dot(c, y, y_dim) - dot(g, z, s_dim);
    for_each_bound_side(input, workspace.bound_sides, num_bound_sides,
                        [&](const int side_index, const int variable_index,
                            const double endpoint, const double jacobian) {
                          duality_gap -= bound_constraint_value(
                                             workspace.vars.x[variable_index],
                                             endpoint, jacobian) *
                                         bound_z[side_index];
                        });
    duality_gap = std::fabs(duality_gap);
    if (std::isnan(duality_gap)) {
      duality_gap = std::numeric_limits<double>::infinity();
    }
  }

  std::copy_n(c, y_dim, ry);
  std::copy_n(gps, s_dim, rz);

  const auto solve_direction = [&](const double target_mu,
                                   const double *affine_ds,
                                   const double *affine_dz,
                                   const double *affine_bound_ds,
                                   const double *affine_bound_dz) {
    std::copy_n(rx, x_dim, bx);
    std::copy_n(c, y_dim, by);
    if (uses_primal_center(settings.mode)) {
      for (int i = 0; i < x_dim; ++i) {
        bx[i] += psi * (workspace.vars.x[i] - centers.x[i]);
      }
    }
    if (uses_dual_center(settings.mode)) {
      for (int i = 0; i < y_dim; ++i) {
        by[i] += r2[i] * (centers.y[i] - y[i]);
      }
    }
    for (int i = 0; i < s_dim; ++i) {
      const double second_order =
          affine_ds == nullptr ? 0.0 : affine_ds[i] * affine_dz[i];
      rs[i] = z[i] - target_mu / s[i] + second_order / s[i];
      bz[i] = rz[i] - w[i] * rs[i];
      if (uses_dual_center(settings.mode)) {
        bz[i] += r3[i] * (centers.z[i] - z[i]);
      }
    }
    for_each_bound_side(
        input, workspace.bound_sides, num_bound_sides,
        [&](const int side_index, const int variable_index, const double,
            const double jacobian) {
          const double second_order =
              affine_bound_ds == nullptr
                  ? 0.0
                  : affine_bound_ds[side_index] * affine_bound_dz[side_index];
          bound_rs[side_index] = bound_z[side_index] -
                                 target_mu / bound_s[side_index] +
                                 second_order / bound_s[side_index];
          const double denominator = bound_w[side_index] + bound_r3[side_index];
          bx[variable_index] += jacobian *
                                (bound_rz[side_index] -
                                 bound_w[side_index] * bound_rs[side_index]) /
                                denominator;
        });
    for (int i = 0; i < dim_3x3; ++i) {
      b[i] = -b[i];
    }
    input.solve(b, v);

    for (int j = 0; j < settings.num_iterative_refinement_steps; ++j) {
      // res = Kv - b; Ku = res; K(v - u) = b.
      double *res_x = residual;
      double *res_y = res_x + x_dim;
      double *res_z = res_y + y_dim;
      for (int i = 0; i < dim_3x3; ++i) {
        residual[i] = -b[i];
      }
      input.add_Kx_to_y(w, r1, r2, r3, vx, vy, vz, res_x, res_y, res_z);
      input.solve(residual, u);
      for (int i = 0; i < dim_3x3; ++i) {
        v[i] -= u[i];
      }
    }

    auto solution = v;
    std::copy_n(solution, x_dim, dx);
    solution += x_dim;
    std::copy_n(solution, y_dim, dy);
    solution += y_dim;
    std::copy_n(solution, s_dim, dz);

    for (int i = 0; i < s_dim; ++i) {
      ds[i] = -w[i] * (dz[i] + rs[i]);
    }
    for_each_bound_side(
        input, workspace.bound_sides, num_bound_sides,
        [&](const int side_index, const int variable_index, const double,
            const double jacobian) {
          const double denominator = bound_w[side_index] + bound_r3[side_index];
          bound_dz[side_index] =
              (jacobian * dx[variable_index] + bound_rz[side_index] -
               bound_w[side_index] * bound_rs[side_index]) /
              denominator;
          bound_ds[side_index] = -bound_w[side_index] *
                                 (bound_dz[side_index] + bound_rs[side_index]);
        });
  };

  const int complementarity_dim = s_dim + num_bound_sides;
  if (settings.barrier.use_predictor_corrector && complementarity_dim > 0) {
    solve_direction(0.0, nullptr, nullptr, nullptr, nullptr);
    double *affine_ds = workspace.next_vars.s;
    double *affine_dz = workspace.next_vars.z;
    double *affine_bound_ds = workspace.next_vars.bound_s;
    double *affine_bound_dz = workspace.next_vars.bound_z;
    std::copy_n(ds, s_dim, affine_ds);
    std::copy_n(dz, s_dim, affine_dz);
    if (num_bound_sides > 0) {
      std::copy_n(bound_ds, num_bound_sides, affine_bound_ds);
      std::copy_n(bound_dz, num_bound_sides, affine_bound_dz);
    }

    const double affine_alpha_s =
        std::min(get_fraction_to_boundary_step(s_dim, 1.0, s, affine_ds),
                 get_fraction_to_boundary_step(num_bound_sides, 1.0, bound_s,
                                               affine_bound_ds));
    const double affine_alpha_z =
        std::min(get_fraction_to_boundary_step(s_dim, 1.0, z, affine_dz),
                 get_fraction_to_boundary_step(num_bound_sides, 1.0, bound_z,
                                               affine_bound_dz));
    double current_mu =
        dot(s, z, s_dim) + dot(bound_s, bound_z, num_bound_sides);
    double affine_mu = 0.0;
    for (int i = 0; i < s_dim; ++i) {
      affine_mu += (s[i] + affine_alpha_s * affine_ds[i]) *
                   (z[i] + affine_alpha_z * affine_dz[i]);
    }
    for (int i = 0; i < num_bound_sides; ++i) {
      affine_mu += (bound_s[i] + affine_alpha_s * affine_bound_ds[i]) *
                   (bound_z[i] + affine_alpha_z * affine_bound_dz[i]);
    }
    current_mu /= complementarity_dim;
    affine_mu /= complementarity_dim;
    const double ratio =
        current_mu > 0.0 ? std::clamp(affine_mu / current_mu, 0.0, 1.0) : 0.0;
    const double sigma = ratio * ratio * ratio;
    solve_direction(sigma * current_mu, affine_ds, affine_dz, affine_bound_ds,
                    affine_bound_dz);
  } else {
    solve_direction(mu, nullptr, nullptr, nullptr, nullptr);
  }
  const auto current_merit_slope =
      merit_slope(input, settings, workspace, mu, psi, dx, ds, dy, dz);

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
  for_each_bound_side(
      input, workspace.bound_sides, num_bound_sides,
      [&](const int side_index, const int, const double, const double) {
        const double abs_gps = std::fabs(bound_gps[side_index]);
        max_constraint_violation =
            std::isnan(abs_gps) ? std::numeric_limits<double>::infinity()
                                : std::max(max_constraint_violation, abs_gps);
        const double sz = bound_s[side_index] * bound_z[side_index];
        max_complementarity = std::isnan(sz)
                                  ? std::numeric_limits<double>::infinity()
                                  : std::max(max_complementarity, sz);
      });

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
      std::min(get_fraction_to_boundary_step(s_dim, tau, workspace.vars.s,
                                             workspace.delta_vars.s),
               get_fraction_to_boundary_step(num_bound_sides, tau,
                                             workspace.vars.bound_s,
                                             workspace.delta_vars.bound_s));
  const double alpha_z_max =
      std::min(get_fraction_to_boundary_step(s_dim, tau, workspace.vars.z,
                                             workspace.delta_vars.z),
               get_fraction_to_boundary_step(num_bound_sides, tau,
                                             workspace.vars.bound_z,
                                             workspace.delta_vars.bound_z));

  if (settings.logging.print_search_direction_logs) {
    double *res_x = residual;
    double *res_s = res_x + x_dim;
    double *res_y = res_s + s_dim;
    double *res_z = res_y + y_dim;
    double *res_bound_reduced = res_z + s_dim;
    double *res_bound_complementarity = res_bound_reduced + num_bound_sides;

    for (int i = 0; i < x_dim; ++i) {
      res_x[i] = -bx[i];
    }

    for (int i = 0; i < y_dim; ++i) {
      res_y[i] = -by[i];
    }

    for (int i = 0; i < s_dim; ++i) {
      res_z[i] = -bz[i];
    }

    input.add_Kx_to_y(w, r1, r2, r3, dx, dy, dz, res_x, res_y, res_z);

    for (int i = 0; i < s_dim; ++i) {
      res_s[i] = ds[i] / w[i] + dz[i] + rs[i];
    }
    for_each_bound_side(input, workspace.bound_sides, num_bound_sides,
                        [&](const int side_index, const int variable_index,
                            const double, const double jacobian) {
                          res_bound_reduced[side_index] =
                              jacobian * dx[variable_index] -
                              (bound_w[side_index] + bound_r3[side_index]) *
                                  bound_dz[side_index] -
                              bound_w[side_index] * bound_rs[side_index] +
                              bound_rz[side_index];
                          res_bound_complementarity[side_index] =
                              bound_ds[side_index] / bound_w[side_index] +
                              bound_dz[side_index] + bound_rs[side_index];
                        });

    lin_sys_error = 0.0;

    const int full_dim = x_dim + y_dim + s_dim + s_dim + 2 * num_bound_sides;
    for (int i = 0; i < full_dim; ++i) {
      lin_sys_error = std::max(lin_sys_error, std::fabs(residual[i]));
    }

    print_search_direction_log_header();

    const MeritSlope observed_merit_slope =
        get_observed_merit_slope(input, settings, mu, psi, tau, workspace);

    fmt::print(fg(fmt::color::green),
               // clang-format off
                   "{:^10} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g}\n",
               // clang-format on
               "", lin_sys_error, alpha_s_max, alpha_z_max,
               current_merit_slope.total, observed_merit_slope.total,
               current_merit_slope.x, observed_merit_slope.x,
               current_merit_slope.s, observed_merit_slope.s,
               current_merit_slope.y, observed_merit_slope.y,
               current_merit_slope.z, observed_merit_slope.z, norm(rx, x_dim),
               std::sqrt(squared_norm(rs, s_dim) +
                         squared_norm(bound_rs, num_bound_sides)));
    const bool suspicious_derivatives =
        (observed_merit_slope.x > 0.0 && current_merit_slope.x < 0.0) ||
        (observed_merit_slope.total > 0.0 && current_merit_slope.total < 0.0) ||
        (std::fabs(current_merit_slope.x - observed_merit_slope.x) /
             std::max({std::fabs(current_merit_slope.x),
                       std::fabs(observed_merit_slope.x), 1e-12}) >
         0.5);
    if (settings.logging.print_derivative_check_logs && suspicious_derivatives)
      check_derivatives(input, settings, tau, workspace);
  }

  return std::make_tuple(
      true, dx, ds, dy, dz, current_merit_slope.total, alpha_s_max, alpha_z_max,
      dual_residual, max_constraint_violation, max_complementarity, kkt_error,
      duality_gap, lin_sys_error, num_regularization_increases);
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
  const int num_bound_sides = workspace.num_bound_sides;

  const auto initial_merit = merit_function(
      input, settings, workspace, workspace.vars.x, workspace.vars.s,
      workspace.vars.y, workspace.vars.z, workspace.vars.bound_s,
      workspace.vars.bound_z, workspace.miscellaneous_workspace.g_plus_s,
      workspace.miscellaneous_workspace.bound_g_plus_s, mu, psi);
  const double m0_f = std::get<0>(initial_merit);
  const double m0 = std::get<5>(initial_merit);

  bool ls_succeeded = false;
  double alpha =
      settings.line_search.start_ls_with_alpha_s_max ? alpha_s_max : 1.0;
  if (uses_primal_center(settings.mode)) {
    alpha = std::min(alpha, alpha_s_max);
  }
  if (uses_dual_center(settings.mode)) {
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
    if (uses_dual_center(settings.mode)) {
      update_next_dual_vars(input, tau, workspace, alpha, true, true);
    }
    const double next_ctc = squared_norm(input.get_c(), y_dim);
    const double next_gsetgse =
        squared_norm(workspace.miscellaneous_workspace.g_plus_s, s_dim);
    const double next_bound_residual_norm = squared_norm(
        workspace.miscellaneous_workspace.bound_g_plus_s, num_bound_sides);
    const double next_sq_constraint_violation_norm =
        next_ctc + next_gsetgse + next_bound_residual_norm;
    const double next_theta = std::sqrt(next_sq_constraint_violation_norm);

    const double *trial_y = uses_dual_center(settings.mode)
                                ? workspace.next_vars.y
                                : workspace.vars.y;
    const double *trial_z = uses_dual_center(settings.mode)
                                ? workspace.next_vars.z
                                : workspace.vars.z;
    const double *trial_bound_z = uses_dual_center(settings.mode)
                                      ? workspace.next_vars.bound_z
                                      : workspace.vars.bound_z;
    const auto trial_merit = merit_function(
        input, settings, workspace, workspace.next_vars.x,
        workspace.next_vars.s, trial_y, trial_z, workspace.next_vars.bound_s,
        trial_bound_z, workspace.miscellaneous_workspace.g_plus_s,
        workspace.miscellaneous_workspace.bound_g_plus_s, mu, psi);
    const double m_f = std::get<0>(trial_merit);
    const double m = std::get<5>(trial_merit);

    merit_delta = m - m0;

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
      double *g_plus_current_s = workspace.csd_workspace.residual;
      double *bound_g_plus_current_s = g_plus_current_s + s_dim;
      for (int i = 0; i < s_dim; ++i) {
        g_plus_current_s[i] = input.get_g()[i] + workspace.vars.s[i];
      }
      set_bound_residuals(input, workspace.next_vars.x, workspace.vars.bound_s,
                          workspace.bound_sides, num_bound_sides,
                          bound_g_plus_current_s);
      const double merit_after_x = std::get<5>(
          merit_function(input, settings, workspace, workspace.next_vars.x,
                         workspace.vars.s, workspace.vars.y, workspace.vars.z,
                         workspace.vars.bound_s, workspace.vars.bound_z,
                         g_plus_current_s, bound_g_plus_current_s, mu, psi));

      const double merit_after_s = std::get<5>(merit_function(
          input, settings, workspace, workspace.next_vars.x,
          workspace.next_vars.s, workspace.vars.y, workspace.vars.z,
          workspace.next_vars.bound_s, workspace.vars.bound_z,
          workspace.miscellaneous_workspace.g_plus_s,
          workspace.miscellaneous_workspace.bound_g_plus_s, mu, psi));
      const double merit_after_y = std::get<5>(merit_function(
          input, settings, workspace, workspace.next_vars.x,
          workspace.next_vars.s, trial_y, workspace.vars.z,
          workspace.next_vars.bound_s, workspace.vars.bound_z,
          workspace.miscellaneous_workspace.g_plus_s,
          workspace.miscellaneous_workspace.bound_g_plus_s, mu, psi));

      const double dm_x = merit_after_x - m0;
      const double dm_s = merit_after_s - merit_after_x;
      const double dm_y = merit_after_y - merit_after_s;
      const double dm_z = m - merit_after_y;

      fmt::print(fg(fmt::color::yellow),
                 // clang-format off
                       "{:^10} {:^+10} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g}\n",
                 // clang-format on
                 "", total_ls_iterations, alpha, m, m_f, std::sqrt(next_ctc),
                 std::sqrt(next_gsetgse + next_bound_residual_norm),
                 merit_delta, merit_delta / alpha, dm_x, dm_s, dm_y, dm_z);
    }

    ++total_ls_iterations;

    if (uses_primal_center(settings.mode)) {
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

  update_next_dual_vars(input, tau, workspace, alpha, true, true);

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

  if (settings.max_iterations < 0 || settings.line_search.max_iterations < 0 ||
      settings.line_search.filter_min_total_line_search_iterations < 0 ||
      settings.num_iterative_refinement_steps < 0 ||
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
  if (uses_primal_center(settings.mode) && settings.line_search.tau == 1.0) {
    return false;
  }
  if (settings.barrier.use_predictor_corrector &&
      !settings.line_search.skip_line_search) {
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

auto check_input(const Input &input, const Workspace &workspace,
                 const int num_bound_sides) -> bool {
  constexpr double infinity = std::numeric_limits<double>::infinity();
  if (input.lower_bounds != nullptr || input.upper_bounds != nullptr) {
    for (int i = 0; i < input.dimensions.x_dim; ++i) {
      const double lower =
          input.lower_bounds == nullptr ? -infinity : input.lower_bounds[i];
      const double upper =
          input.upper_bounds == nullptr ? infinity : input.upper_bounds[i];
      if (std::isnan(lower) || std::isnan(upper) || lower >= upper) {
        return false;
      }
    }
  }
  for (int i = 0; i < num_bound_sides; ++i) {
    if (!std::isfinite(workspace.vars.bound_s[i]) ||
        workspace.vars.bound_s[i] <= 0.0 ||
        !std::isfinite(workspace.vars.bound_z[i]) ||
        workspace.vars.bound_z[i] <= 0.0) {
      return false;
    }
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

  const int x_dim = input.dimensions.x_dim;
  const int s_dim = input.dimensions.s_dim;
  const int y_dim = input.dimensions.y_dim;
  workspace.num_bound_sides =
      initialize_bound_sides(input, workspace.bound_sides);
  const int num_bound_sides = workspace.num_bound_sides;

  if (!check_settings(settings) ||
      !check_input(input, workspace, num_bound_sides)) {
    if (settings.assert_checks_pass) {
      assert(false && "check_settings or check_input returned false.");
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

  std::copy_n(workspace.vars.x, x_dim, workspace.proximal_centers.x);
  std::copy_n(workspace.vars.y, y_dim, workspace.proximal_centers.y);
  std::copy_n(workspace.vars.z, s_dim, workspace.proximal_centers.z);
  std::copy_n(workspace.vars.bound_z, num_bound_sides,
              workspace.proximal_centers.bound_z);

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
  set_bound_residuals(input, workspace.vars.x, workspace.vars.bound_s,
                      workspace.bound_sides, num_bound_sides,
                      workspace.miscellaneous_workspace.bound_g_plus_s);

  if (!settings.penalty.warm_start_penalties) {
    std::fill_n(workspace.penalties.y, y_dim,
                settings.penalty.initial_penalty_parameter);
    std::fill_n(workspace.penalties.z, s_dim,
                settings.penalty.initial_penalty_parameter);
    if (num_bound_sides > 0) {
      std::fill_n(workspace.penalties.bound_z, num_bound_sides,
                  settings.penalty.initial_penalty_parameter);
    }
  }

  double mu = settings.barrier.initial_mu;
  double psi = uses_primal_center(settings.mode)
                   ? std::max(settings.regularization.initial,
                              settings.regularization.first_positive)
                   : settings.regularization.initial;
  const double tau = settings.line_search.tau;

  int total_ls_iterations = 0;
  // A single poorly conditioned solve can produce a spurious flat direction.
  int num_consecutive_stalled_iterations = 0;
  CenterUpdateRejections primal_center_update_rejections;
  CenterUpdateRejections dual_center_update_rejections;
  std::optional<double> previous_cost;
  workspace.filter.size = 0;

  for (int iteration = 0; iteration < settings.max_iterations; ++iteration) {
    const double f0 = input.get_f();

    const double ctc = squared_norm(input.get_c(), y_dim);
    const double gsetgse =
        squared_norm(workspace.miscellaneous_workspace.g_plus_s, s_dim);
    const double bound_gsetgse = squared_norm(
        workspace.miscellaneous_workspace.bound_g_plus_s, num_bound_sides);

    const double sq_constraint_violation_norm = ctc + gsetgse + bound_gsetgse;

    const auto [factorization_ok, dx, ds, dy, dz, merit_slope, alpha_s_max,
                alpha_z_max, dual_residual, max_constraint_violation,
                max_complementarity, kkt_error, duality_gap, lin_sys_error,
                num_regularization_increases] =
        compute_search_direction(input, settings, mu, psi, tau, workspace);

    const double inequality_residual_norm = std::sqrt(gsetgse + bound_gsetgse);
    const double ds_norm =
        std::sqrt(squared_norm(ds, s_dim) +
                  squared_norm(workspace.delta_vars.bound_s, num_bound_sides));
    const double dz_norm =
        std::sqrt(squared_norm(dz, s_dim) +
                  squared_norm(workspace.delta_vars.bound_z, num_bound_sides));

    if (!factorization_ok) {
      return Output{
          .exit_status = Status::FACTORIZATION_FAILURE,
          .num_iterations = iteration,
          .num_ls_iterations = total_ls_iterations,
          .max_primal_violation = max_primal_violation(
              input, workspace, num_bound_sides, workspace.vars.x),
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
      const double eta =
          mean_penalty_parameter(workspace, s_dim, y_dim, num_bound_sides);
      fmt::print(
          fg(fmt::color::red),
          // clang-format off
                       "{:^+10} {:^10} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^10} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^10} {:^+10.4g} {:^+10.4g} {:^+10.4g}\n",
          // clang-format on
          iteration, "", input.get_f(), std::sqrt(ctc),
          inequality_residual_norm, "", norm(dx, x_dim), ds_norm,
          norm(dy, y_dim), dz_norm, mu, eta, tau, psi,
          num_regularization_increases, max_complementarity, dual_residual,
          kkt_error);
    }

    if (termination.stalled) {
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
          .max_primal_violation = max_primal_violation(
              input, workspace, num_bound_sides, workspace.vars.x),
          .max_dual_violation = inf_norm(workspace.nrhs.x, x_dim),
      };
    }

    if (termination.advance_barrier) {
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
          .max_primal_violation = max_primal_violation(
              input, workspace, num_bound_sides, workspace.vars.x),
          .max_dual_violation = inf_norm(workspace.nrhs.x, x_dim),
      };
    }

    bool ls_succeeded;
    double alpha, m0;
    if (settings.line_search.skip_line_search) {
      if (settings.barrier.use_predictor_corrector) {
        alpha = alpha_s_max;
        update_next_primal_vars(input, tau, workspace, alpha_s_max, true, true);
        update_next_dual_vars(input, tau, workspace, alpha_z_max, true, true);
      } else {
        alpha = uses_dual_center(settings.mode)
                    ? std::min(alpha_s_max, alpha_z_max)
                    : alpha_s_max;
        update_next_primal_vars(input, tau, workspace, alpha, true, true);
        update_next_dual_vars(input, tau, workspace, alpha, true, true);
      }
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

      const double eta =
          mean_penalty_parameter(workspace, s_dim, y_dim, num_bound_sides);
      fmt::print(
          fg(fmt::color::red),
          // clang-format off
                       "{:^+10} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g}  {:^+10.4g} {:^+10.4g} {:^10} {:^+10.4g} {:^+10.4g} {:^+10.4g}\n",
          // clang-format on
          iteration, alpha, f0, std::sqrt(ctc), inequality_residual_norm, m0,
          norm(dx, x_dim), ds_norm, norm(dy, y_dim), dz_norm, mu, eta, tau, psi,
          num_regularization_increases, max_complementarity, dual_residual,
          kkt_error);
    }

    if (settings.line_search.enable_line_search_failures && !ls_succeeded) {
      return Output{
          .exit_status = Status::LINE_SEARCH_FAILURE,
          .num_iterations = iteration,
          .num_ls_iterations = total_ls_iterations,
          .max_primal_violation = max_primal_violation(
              input, workspace, num_bound_sides, workspace.vars.x),
          .max_dual_violation = inf_norm(workspace.nrhs.x, x_dim),
      };
    }

    const double previous_complementarity =
        mean_complementarity(workspace, s_dim, num_bound_sides);
    std::swap(workspace.vars, workspace.next_vars);
    previous_cost = f0;

    if (input.timeout_callback()) {
      return Output{
          .exit_status = Status::TIMEOUT,
          .num_iterations = iteration,
          .num_ls_iterations = total_ls_iterations,
          .max_primal_violation = max_primal_violation(
              input, workspace, num_bound_sides, workspace.vars.x),
          .max_dual_violation = inf_norm(workspace.nrhs.x, x_dim),
      };
    }

    if (kkt_error <= settings.barrier.mu_update_kappa * mu) {
      mu = std::max(mu * settings.barrier.mu_update_factor,
                    settings.barrier.mu_min);
    }
    if (settings.barrier.use_predictor_corrector &&
        uses_primal_center(settings.mode) && s_dim + num_bound_sides > 0) {
      const double current_complementarity =
          mean_complementarity(workspace, s_dim, num_bound_sides);
      const double complementarity_ratio =
          previous_complementarity > 0.0
              ? std::clamp(current_complementarity / previous_complementarity,
                           0.0, 1.0)
              : 1.0;
      constexpr double residual_reduction_factor = 0.95;
      constexpr double proximal_residual_threshold = 0.9;
      constexpr double inexact_residual_ratio = 0.1;
      constexpr double rejected_reduction_scale = 0.666;
      constexpr int startup_iterations = 5;
      constexpr int max_update_rejections = 8;

      const auto [new_primal_residual, new_dual_residual] =
          unregularized_residuals(input, workspace);
      double dual_proximal_residual = 0.0;
      for (int i = 0; i < x_dim; ++i) {
        const double displacement =
            workspace.vars.x[i] - workspace.proximal_centers.x[i];
        dual_proximal_residual =
            std::max(dual_proximal_residual, std::fabs(psi * displacement));
        workspace.nrhs.x[i] += psi * displacement;
      }
      const double regularized_dual_residual =
          max_abs_or_inf(workspace.nrhs.x, x_dim);

      const bool primal_center_accepted =
          new_dual_residual < residual_reduction_factor * dual_residual ||
          new_dual_residual < settings.termination.max_dual_residual;
      if (primal_center_accepted) {
        std::copy_n(workspace.vars.x, x_dim, workspace.proximal_centers.x);
        primal_center_update_rejections.reset();
        psi = std::max(settings.regularization.first_positive,
                       complementarity_ratio * psi);
      } else {
        primal_center_update_rejections.reject(dual_residual);
        if (iteration < startup_iterations ||
            dual_proximal_residual < proximal_residual_threshold) {
          const double factor =
              1.0 - rejected_reduction_scale * (1.0 - complementarity_ratio);
          psi = std::max(settings.regularization.first_positive, factor * psi);
        }
        if (primal_center_update_rejections.has_reduced_residual(
                new_dual_residual, residual_reduction_factor)) {
          primal_center_update_rejections.reset();
        } else if (primal_center_update_rejections.count >=
                       max_update_rejections &&
                   dual_proximal_residual >= proximal_residual_threshold &&
                   regularized_dual_residual <
                       std::max(settings.termination.max_dual_residual,
                                inexact_residual_ratio * new_dual_residual)) {
          std::copy_n(workspace.vars.x, x_dim, workspace.proximal_centers.x);
          primal_center_update_rejections.reset();
        }
      }

      if (uses_dual_center(settings.mode)) {
        double primal_proximal_residual = 0.0;
        double regularized_primal_residual = 0.0;
        const auto update_residuals =
            [&](const double *residual, const double *value,
                const double *center, const double *penalties, const int size) {
              for (int i = 0; i < size; ++i) {
                const double proximal = (center[i] - value[i]) / penalties[i];
                primal_proximal_residual =
                    std::max(primal_proximal_residual, std::fabs(proximal));
                regularized_primal_residual =
                    std::max(regularized_primal_residual,
                             std::fabs(residual[i] + proximal));
              }
            };
        update_residuals(input.get_c(), workspace.vars.y,
                         workspace.proximal_centers.y, workspace.penalties.y,
                         y_dim);
        update_residuals(workspace.miscellaneous_workspace.g_plus_s,
                         workspace.vars.z, workspace.proximal_centers.z,
                         workspace.penalties.z, s_dim);
        update_residuals(workspace.miscellaneous_workspace.bound_g_plus_s,
                         workspace.vars.bound_z,
                         workspace.proximal_centers.bound_z,
                         workspace.penalties.bound_z, num_bound_sides);

        const bool dual_center_accepted =
            new_primal_residual <
                residual_reduction_factor * max_constraint_violation ||
            new_primal_residual < settings.termination.max_constraint_violation;
        double regularization_factor = 1.0;
        if (dual_center_accepted) {
          std::copy_n(workspace.vars.y, y_dim, workspace.proximal_centers.y);
          std::copy_n(workspace.vars.z, s_dim, workspace.proximal_centers.z);
          std::copy_n(workspace.vars.bound_z, num_bound_sides,
                      workspace.proximal_centers.bound_z);
          dual_center_update_rejections.reset();
          regularization_factor = complementarity_ratio;
        } else {
          dual_center_update_rejections.reject(max_constraint_violation);
          if (iteration < startup_iterations ||
              primal_proximal_residual < proximal_residual_threshold) {
            regularization_factor =
                1.0 - rejected_reduction_scale * (1.0 - complementarity_ratio);
          }
          if (dual_center_update_rejections.has_reduced_residual(
                  new_primal_residual, residual_reduction_factor)) {
            dual_center_update_rejections.reset();
          } else if (dual_center_update_rejections.count >=
                         max_update_rejections &&
                     primal_proximal_residual >= proximal_residual_threshold &&
                     regularized_primal_residual <
                         std::max(settings.termination.max_constraint_violation,
                                  inexact_residual_ratio *
                                      new_primal_residual)) {
            std::copy_n(workspace.vars.y, y_dim, workspace.proximal_centers.y);
            std::copy_n(workspace.vars.z, s_dim, workspace.proximal_centers.z);
            std::copy_n(workspace.vars.bound_z, num_bound_sides,
                        workspace.proximal_centers.bound_z);
            dual_center_update_rejections.reset();
          }
        }

        const auto update_penalties = [&](double *penalties, const int size) {
          for (int i = 0; i < size; ++i) {
            penalties[i] =
                regularization_factor > 0.0
                    ? std::min(settings.penalty.max_penalty_parameter,
                               penalties[i] / regularization_factor)
                    : settings.penalty.max_penalty_parameter;
          }
        };
        update_penalties(workspace.penalties.y, y_dim);
        update_penalties(workspace.penalties.z, s_dim);
        update_penalties(workspace.penalties.bound_z, num_bound_sides);
      } else {
        update_penalty_parameters(input, settings, workspace);
      }
    } else {
      psi = decreased_regularization(settings, psi);

      const bool any_penalty_increased =
          update_penalty_parameters(input, settings, workspace);
      if (any_penalty_increased) {
        // Reset regularization when penalty increases, to stabilize the
        // modified KKT system.
        psi = std::max(psi, settings.regularization.initial);
      }
    }
  }

  return Output{
      .exit_status = Status::ITERATION_LIMIT,
      .num_iterations = settings.max_iterations,
      .num_ls_iterations = total_ls_iterations,
      .max_primal_violation = max_primal_violation(
          input, workspace, num_bound_sides, workspace.vars.x),
      .max_dual_violation = inf_norm(workspace.nrhs.x, x_dim),
  };
}

} // namespace sip
