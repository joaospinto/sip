#include "sip/sip.hpp"

#include <gtest/gtest.h>

namespace sip {

TEST(SimpleNLP, SYMMETRIC_DIRECT_4x4) {
  Input input;
  Settings settings{.max_kkt_violation = 1e-12,
                    .lin_sys_formulation =
                        Settings::LinearSystemFormulation::SYMMETRIC_DIRECT_4x4,
                    .permute_kkt_system = true,
                    .enable_elastics = true,
                    .elastic_var_cost_coeff = 1e6};
  Workspace workspace;
  Output output;

  constexpr int x_dim = 2;
  constexpr int s_dim = 2;
  constexpr int y_dim = 0;
  constexpr int upper_hessian_f_nnz = 3;
  constexpr int jacobian_c_nnz = 0;
  constexpr int jacobian_g_nnz = 4;
  constexpr int _unused_upper_jac_g_t_jac_g_nnz = 0;
  constexpr int _unused_upper_hessian_f_plus_upper_jac_g_t_jac_g_nnz = 0;
  constexpr int kkt_L_nnz = 11;

  workspace.reserve(
      settings.lin_sys_formulation, settings.custom_lin_sys_solver.has_value(),
      x_dim, s_dim, y_dim, upper_hessian_f_nnz, jacobian_c_nnz, jacobian_g_nnz,
      _unused_upper_jac_g_t_jac_g_nnz,
      _unused_upper_hessian_f_plus_upper_jac_g_t_jac_g_nnz, kkt_L_nnz);

  auto model_callback = [](const ModelCallbackInput &mci,
                           ModelCallbackOutput &mco) -> void {
    mco.f = mci.x[1] * (5.0 + mci.x[0]);

    mco.gradient_f[0] = mci.x[1];
    mco.gradient_f[1] = 5.0 + mci.x[0];

    // NOTE: a positive definite Hessian approximation is expected.
    mco.upper_hessian_f.rows = x_dim;
    mco.upper_hessian_f.cols = x_dim;
    mco.upper_hessian_f.ind[0] = 0;
    mco.upper_hessian_f.ind[1] = 0;
    mco.upper_hessian_f.ind[2] = 1;
    mco.upper_hessian_f.indptr[0] = 0;
    mco.upper_hessian_f.indptr[1] = 1;
    mco.upper_hessian_f.indptr[2] = 3;
    // NOTE: only the upper triangle should be filled.
    //       the eigenvalues of the real Hessian are +-1,
    //       so we add (1 + 1e-6) to shift them.
    mco.upper_hessian_f.data[0] = 1.0 + 1e-6;
    mco.upper_hessian_f.data[1] = 1.0;
    mco.upper_hessian_f.data[2] = 1.0 + 1e-6;
    mco.upper_hessian_f.is_transposed = false;

    // No equality constraints, so we don't set mco.c.

    mco.jacobian_c.rows = x_dim;
    mco.jacobian_c.cols = y_dim;
    mco.jacobian_c.indptr[0] = 0;
    mco.jacobian_c.is_transposed = true;

    mco.g[0] = 5.0 - mci.x[0] * mci.x[1];
    mco.g[1] = mci.x[0] * mci.x[0] + mci.x[1] * mci.x[1] - 20.0;

    mco.jacobian_g.rows = x_dim;
    mco.jacobian_g.cols = s_dim;
    mco.jacobian_g.ind[0] = 0;
    mco.jacobian_g.ind[1] = 1;
    mco.jacobian_g.ind[2] = 0;
    mco.jacobian_g.ind[3] = 1;
    mco.jacobian_g.indptr[0] = 0;
    mco.jacobian_g.indptr[1] = 2;
    mco.jacobian_g.indptr[2] = 4;
    mco.jacobian_g.data[0] = -mci.x[1];
    mco.jacobian_g.data[1] = -mci.x[0];
    mco.jacobian_g.data[2] = 2 * mci.x[0];
    mco.jacobian_g.data[3] = 2 * mci.x[1];
    mco.jacobian_g.is_transposed = true;
  };

  input.model_callback = model_callback;

  const auto kkt_p = std::array{5, 4, 3, 2, 1, 0};
  const auto kkt_pinv = std::array{5, 4, 3, 2, 1, 0};
  input.kkt_p = kkt_p.data();
  input.kkt_pinv = kkt_pinv.data();

  for (int i = 0; i < x_dim; ++i) {
    workspace.vars.x[i] = 0.0;
  }

  for (int i = 0; i < s_dim; ++i) {
    workspace.vars.s[i] = 1.0;
    workspace.vars.z[i] = 1.0;
    workspace.vars.e[i] = 0.0;
  }

  for (int i = 0; i < y_dim; ++i) {
    workspace.vars.y[i] = 0.0;
  }

  solve(input, settings, workspace, output);

  EXPECT_EQ(output.exit_status, Status::SOLVED);

  EXPECT_NEAR(workspace.vars.x[0], -1.15747396, 1e-6);
  EXPECT_NEAR(workspace.vars.x[1], -4.31975162, 1e-6);

  workspace.free();
}

TEST(SimpleNLP, SYMMETRIC_INDIRECT_3x3) {
  Input input;
  Settings settings{
      .max_kkt_violation = 1e-12,
      .lin_sys_formulation =
          Settings::LinearSystemFormulation::SYMMETRIC_INDIRECT_3x3,
      .permute_kkt_system = true,
      .enable_elastics = true,
      .elastic_var_cost_coeff = 1e6};
  Workspace workspace;
  Output output;

  constexpr int x_dim = 2;
  constexpr int s_dim = 2;
  constexpr int y_dim = 0;
  constexpr int upper_hessian_f_nnz = 3;
  constexpr int jacobian_c_nnz = 0;
  constexpr int jacobian_g_nnz = 4;
  constexpr int _unused_upper_jac_g_t_jac_g_nnz = 0;
  constexpr int _unused_upper_hessian_f_plus_upper_jac_g_t_jac_g_nnz = 0;
  constexpr int kkt_L_nnz = 5;

  workspace.reserve(
      settings.lin_sys_formulation, settings.custom_lin_sys_solver.has_value(),
      x_dim, s_dim, y_dim, upper_hessian_f_nnz, jacobian_c_nnz, jacobian_g_nnz,
      _unused_upper_jac_g_t_jac_g_nnz,
      _unused_upper_hessian_f_plus_upper_jac_g_t_jac_g_nnz, kkt_L_nnz);

  auto model_callback = [](const ModelCallbackInput &mci,
                           ModelCallbackOutput &mco) -> void {
    mco.f = mci.x[1] * (5.0 + mci.x[0]);

    mco.gradient_f[0] = mci.x[1];
    mco.gradient_f[1] = 5.0 + mci.x[0];

    // NOTE: a positive definite Hessian approximation is expected.
    mco.upper_hessian_f.rows = x_dim;
    mco.upper_hessian_f.cols = x_dim;
    mco.upper_hessian_f.ind[0] = 0;
    mco.upper_hessian_f.ind[1] = 0;
    mco.upper_hessian_f.ind[2] = 1;
    mco.upper_hessian_f.indptr[0] = 0;
    mco.upper_hessian_f.indptr[1] = 1;
    mco.upper_hessian_f.indptr[2] = 3;
    // NOTE: only the upper triangle should be filled.
    //       the eigenvalues of the real Hessian are +-1,
    //       so we add (1 + 1e-6) to shift them.
    mco.upper_hessian_f.data[0] = 1.0 + 1e-6;
    mco.upper_hessian_f.data[1] = 1.0;
    mco.upper_hessian_f.data[2] = 1.0 + 1e-6;
    mco.upper_hessian_f.is_transposed = false;

    // No equality constraints, so we don't set mco.c.

    mco.jacobian_c.rows = x_dim;
    mco.jacobian_c.cols = y_dim;
    mco.jacobian_c.indptr[0] = 0;
    mco.jacobian_c.is_transposed = true;

    mco.g[0] = 5.0 - mci.x[0] * mci.x[1];
    mco.g[1] = mci.x[0] * mci.x[0] + mci.x[1] * mci.x[1] - 20.0;

    mco.jacobian_g.rows = x_dim;
    mco.jacobian_g.cols = s_dim;
    mco.jacobian_g.ind[0] = 0;
    mco.jacobian_g.ind[1] = 1;
    mco.jacobian_g.ind[2] = 0;
    mco.jacobian_g.ind[3] = 1;
    mco.jacobian_g.indptr[0] = 0;
    mco.jacobian_g.indptr[1] = 2;
    mco.jacobian_g.indptr[2] = 4;
    mco.jacobian_g.data[0] = -mci.x[1];
    mco.jacobian_g.data[1] = -mci.x[0];
    mco.jacobian_g.data[2] = 2 * mci.x[0];
    mco.jacobian_g.data[3] = 2 * mci.x[1];
    mco.jacobian_g.is_transposed = true;
  };

  input.model_callback = model_callback;

  const auto kkt_p = std::array{3, 2, 1, 0};
  const auto kkt_pinv = std::array{3, 2, 1, 0};
  input.kkt_p = kkt_p.data();
  input.kkt_pinv = kkt_pinv.data();

  for (int i = 0; i < x_dim; ++i) {
    workspace.vars.x[i] = 0.0;
  }

  for (int i = 0; i < s_dim; ++i) {
    workspace.vars.s[i] = 1.0;
    workspace.vars.z[i] = 1.0;
    workspace.vars.e[i] = 0.0;
  }

  for (int i = 0; i < y_dim; ++i) {
    workspace.vars.y[i] = 0.0;
  }

  solve(input, settings, workspace, output);

  EXPECT_EQ(output.exit_status, Status::SOLVED);

  EXPECT_NEAR(workspace.vars.x[0], -1.15747396, 1e-6);
  EXPECT_NEAR(workspace.vars.x[1], -4.31975162, 1e-6);

  workspace.free();
}

TEST(SimpleNLP, SYMMETRIC_INDIRECT_2x2) {
  Input input;
  Settings settings{
      .max_kkt_violation = 1e-12,
      .lin_sys_formulation =
          Settings::LinearSystemFormulation::SYMMETRIC_INDIRECT_2x2,
      .enable_elastics = true,
      .elastic_var_cost_coeff = 1e6};
  Workspace workspace;
  Output output;

  constexpr int x_dim = 2;
  constexpr int s_dim = 2;
  constexpr int y_dim = 0;
  constexpr int upper_hessian_f_nnz = 3;
  constexpr int jacobian_c_nnz = 0;
  constexpr int jacobian_g_nnz = 4;
  constexpr int upper_jac_g_t_jac_g_nnz = 3;
  constexpr int upper_hessian_f_plus_upper_jac_g_t_jac_g_nnz = 3;
  constexpr int kkt_L_nnz = 1;

  workspace.reserve(settings.lin_sys_formulation,
                    settings.custom_lin_sys_solver.has_value(), x_dim, s_dim,
                    y_dim, upper_hessian_f_nnz, jacobian_c_nnz, jacobian_g_nnz,
                    upper_jac_g_t_jac_g_nnz,
                    upper_hessian_f_plus_upper_jac_g_t_jac_g_nnz, kkt_L_nnz);

  auto model_callback = [](const ModelCallbackInput &mci,
                           ModelCallbackOutput &mco) -> void {
    mco.f = mci.x[1] * (5.0 + mci.x[0]);

    mco.gradient_f[0] = mci.x[1];
    mco.gradient_f[1] = 5.0 + mci.x[0];

    // NOTE: a positive definite Hessian approximation is expected.
    mco.upper_hessian_f.rows = x_dim;
    mco.upper_hessian_f.cols = x_dim;
    mco.upper_hessian_f.ind[0] = 0;
    mco.upper_hessian_f.ind[1] = 0;
    mco.upper_hessian_f.ind[2] = 1;
    mco.upper_hessian_f.indptr[0] = 0;
    mco.upper_hessian_f.indptr[1] = 1;
    mco.upper_hessian_f.indptr[2] = 3;
    // NOTE: only the upper triangle should be filled.
    //       the eigenvalues of the real Hessian are +-1,
    //       so we add (1 + 1e-6) to shift them.
    mco.upper_hessian_f.data[0] = 1.0 + 1e-6;
    mco.upper_hessian_f.data[1] = 1.0;
    mco.upper_hessian_f.data[2] = 1.0 + 1e-6;
    mco.upper_hessian_f.is_transposed = false;

    // No equality constraints, so we don't set mco.c.

    mco.jacobian_c.rows = x_dim;
    mco.jacobian_c.cols = y_dim;
    mco.jacobian_c.indptr[0] = 0;
    mco.jacobian_c.is_transposed = true;

    mco.g[0] = 5.0 - mci.x[0] * mci.x[1];
    mco.g[1] = mci.x[0] * mci.x[0] + mci.x[1] * mci.x[1] - 20.0;

    mco.jacobian_g.rows = s_dim;
    mco.jacobian_g.cols = x_dim;
    mco.jacobian_g.ind[0] = 0;
    mco.jacobian_g.ind[1] = 1;
    mco.jacobian_g.ind[2] = 0;
    mco.jacobian_g.ind[3] = 1;
    mco.jacobian_g.indptr[0] = 0;
    mco.jacobian_g.indptr[1] = 2;
    mco.jacobian_g.indptr[2] = 4;
    mco.jacobian_g.data[0] = -mci.x[1];
    mco.jacobian_g.data[1] = 2 * mci.x[0];
    mco.jacobian_g.data[2] = -mci.x[0];
    mco.jacobian_g.data[3] = 2 * mci.x[1];
    mco.jacobian_g.is_transposed = false;
  };

  input.model_callback = model_callback;

  const auto kkt_p = std::array{1, 0};
  const auto kkt_pinv = std::array{1, 0};
  input.kkt_p = kkt_p.data();
  input.kkt_pinv = kkt_pinv.data();

  for (int i = 0; i < x_dim; ++i) {
    workspace.vars.x[i] = 0.0;
  }

  for (int i = 0; i < s_dim; ++i) {
    workspace.vars.s[i] = 1.0;
    workspace.vars.z[i] = 1.0;
    workspace.vars.e[i] = 0.0;
  }

  for (int i = 0; i < y_dim; ++i) {
    workspace.vars.y[i] = 0.0;
  }

  solve(input, settings, workspace, output);

  EXPECT_EQ(output.exit_status, Status::SOLVED);

  EXPECT_NEAR(workspace.vars.x[0], -1.15747396, 1e-6);
  EXPECT_NEAR(workspace.vars.x[1], -4.31975162, 1e-6);

  workspace.free();
}

} // namespace sip
