#include "sip/sip.hpp"

#include <cmath>

namespace {

auto test_recovers_from_one_stalled_direction() -> bool {
  double objective = 0.0;
  double gradient = 0.0;
  double *model_x = nullptr;
  double regularization = 0.0;
  int solve_count = 0;

  const auto model_callback = [&](const sip::ModelCallbackInput &input) {
    model_x = input.x;
    objective = 0.5 * (model_x[0] - 1.0) * (model_x[0] - 1.0);
    gradient = model_x[0] - 1.0;
  };
  const auto factor = [&](const double *, const double r1, const double *,
                          const double *) {
    regularization = r1;
    return true;
  };
  const auto inexact_solve = [&](const double *b, double *v) {
    ++solve_count;
    // Model one transient loss of accuracy from an ill-conditioned solve.
    v[0] = solve_count == 1 ? 0.0 : b[0] / (1.0 + regularization);
  };
  const auto add_kx_to_y = [&](const double *, const double r1, const double *,
                               const double *, const double *x, const double *,
                               const double *, double *y, double *,
                               double *) { y[0] += (1.0 + r1) * x[0]; };
  const auto add_hx_to_y = [](const double *x, double *y) { y[0] += x[0]; };
  const auto no_op = [](const double *, double *) {};

  sip::Settings settings;
  settings.max_iterations = 4;
  settings.num_iterative_refinement_steps = 0;
  settings.termination.max_dual_residual = 1e-12;
  settings.line_search.skip_line_search = true;
  settings.logging.print_logs = false;
  settings.logging.print_line_search_logs = false;
  settings.logging.print_search_direction_logs = false;
  settings.logging.print_derivative_check_logs = false;

  sip::Input input{
      .factor = factor,
      .solve = inexact_solve,
      .add_Kx_to_y = add_kx_to_y,
      .add_Hx_to_y = add_hx_to_y,
      .add_Cx_to_y = no_op,
      .add_CTx_to_y = no_op,
      .add_Gx_to_y = no_op,
      .add_GTx_to_y = no_op,
      .get_f = [&] { return objective; },
      .get_grad_f = [&] { return &gradient; },
      .get_c = [] { return static_cast<const double *>(nullptr); },
      .get_g = [] { return static_cast<const double *>(nullptr); },
      .model_callback = model_callback,
      .timeout_callback = [] { return false; },
      .dimensions = {.x_dim = 1, .s_dim = 0, .y_dim = 0},
  };

  sip::Workspace workspace;
  workspace.reserve(1, 0, 0, settings);
  workspace.vars.x[0] = 0.0;

  const sip::Output output = sip::solve(input, settings, workspace);
  const bool passed = output.exit_status == sip::Status::SOLVED &&
                      output.num_iterations == 2 && solve_count == 3 &&
                      std::abs(workspace.vars.x[0] - 1.0) < 1e-12;
  workspace.free();
  return passed;
}

} // namespace

auto main() -> int {
  return test_recovers_from_one_stalled_direction() ? 0 : 1;
}
