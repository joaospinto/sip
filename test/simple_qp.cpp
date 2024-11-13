#include "sip/sip.hpp"

#include <gtest/gtest.h>

namespace sip {

TEST(SimpleQPFromOSQPRepo, SYMMETRIC_DIRECT_4x4) {
  Input input;
  Settings settings{
      .max_kkt_violation = 1e-12,
      .lin_sys_formulation =
          Settings::LinearSystemFormulation::SYMMETRIC_DIRECT_4x4};
  Workspace workspace;
  Output output;

  constexpr int x_dim = 2;
  constexpr int s_dim = 4;
  constexpr int y_dim = 1;
  constexpr int upper_hessian_f_nnz = 3;
  constexpr int jacobian_c_nnz = 2;
  constexpr int jacobian_g_nnz = 4;
  constexpr int upper_jac_g_t_jac_g_nnz =
      0; // Unused with SYMMETRIC_DIRECT_4x4.
  constexpr int upper_hessian_f_plus_upper_jac_g_t_jac_g_nnz =
      0; // Unused with SYMMETRIC_DIRECT_4x4.
  constexpr int kkt_L_nnz = 23;

  workspace.reserve(settings.lin_sys_formulation, x_dim, s_dim, y_dim,
                    upper_hessian_f_nnz, jacobian_c_nnz, jacobian_g_nnz,
                    upper_jac_g_t_jac_g_nnz,
                    upper_hessian_f_plus_upper_jac_g_t_jac_g_nnz, kkt_L_nnz);

  auto model_callback = [](const ModelCallbackInput &mci,
                           ModelCallbackOutput &mco) -> void {
    mco.f = 0.5 * (4.0 * mci.x[0] * mci.x[0] + 2.0 * mci.x[0] * mci.x[1] +
                   2.0 * mci.x[1] * mci.x[1]) +
            mci.x[0] + mci.x[1];

    mco.gradient_f[0] = 4.0 * mci.x[0] + 1.0 * mci.x[1] + 1.0;
    mco.gradient_f[1] = 1.0 * mci.x[0] + 2.0 * mci.x[1] + 1.0;

    // NOTE: only the upper triangle should be filled.
    mco.upper_hessian_f.rows = x_dim;
    mco.upper_hessian_f.cols = x_dim;
    mco.upper_hessian_f.ind[0] = 0;
    mco.upper_hessian_f.ind[1] = 0;
    mco.upper_hessian_f.ind[2] = 1;
    mco.upper_hessian_f.indptr[0] = 0;
    mco.upper_hessian_f.indptr[1] = 1;
    mco.upper_hessian_f.indptr[2] = 3;
    mco.upper_hessian_f.data[0] = 4.0;
    mco.upper_hessian_f.data[1] = 1.0;
    mco.upper_hessian_f.data[2] = 2.0;

    mco.c[0] = mci.x[0] + mci.x[1] - 1.0;

    mco.jacobian_c.rows = x_dim;
    mco.jacobian_c.cols = y_dim;
    mco.jacobian_c.ind[0] = 0;
    mco.jacobian_c.ind[1] = 1;
    mco.jacobian_c.indptr[0] = 0;
    mco.jacobian_c.indptr[1] = 2;
    mco.jacobian_c.data[0] = 1.0;
    mco.jacobian_c.data[1] = 1.0;
    mco.jacobian_c.is_transposed = true;

    mco.g[0] = mci.x[0] - 0.7;
    mco.g[1] = -mci.x[0] - 0.0;
    mco.g[2] = mci.x[1] - 0.7;
    mco.g[3] = -mci.x[1] - 0.0;

    mco.jacobian_g.rows = x_dim;
    mco.jacobian_g.cols = s_dim;
    mco.jacobian_g.ind[0] = 0;
    mco.jacobian_g.ind[1] = 0;
    mco.jacobian_g.ind[2] = 1;
    mco.jacobian_g.ind[3] = 1;
    mco.jacobian_g.indptr[0] = 0;
    mco.jacobian_g.indptr[1] = 1;
    mco.jacobian_g.indptr[2] = 2;
    mco.jacobian_g.indptr[3] = 3;
    mco.jacobian_g.indptr[4] = 4;
    mco.jacobian_g.data[0] = 1.0;
    mco.jacobian_g.data[1] = -1.0;
    mco.jacobian_g.data[2] = 1.0;
    mco.jacobian_g.data[3] = -1.0;
    mco.jacobian_g.is_transposed = true;
  };

  input.model_callback = model_callback;

  for (int i = 0; i < x_dim; ++i) {
    workspace.vars.x[i] = 0.0;
  }

  for (int i = 0; i < s_dim; ++i) {
    workspace.vars.s[i] = 1.0;
    workspace.vars.z[i] = 1.0;
  }

  for (int i = 0; i < y_dim; ++i) {
    workspace.vars.y[i] = 0.0;
  }

  solve(input, settings, workspace, output);

  EXPECT_EQ(output.exit_status, Status::SOLVED);

  EXPECT_NEAR(workspace.vars.x[0], 0.3, 1e-6);
  EXPECT_NEAR(workspace.vars.x[1], 0.7, 1e-6);

  workspace.free();
}

TEST(SimpleQPFromOSQPRepo, SYMMETRIC_INDIRECT_2x2) {
  Input input;
  Settings settings{
      .max_kkt_violation = 1e-12,
      .lin_sys_formulation =
          Settings::LinearSystemFormulation::SYMMETRIC_INDIRECT_2x2};
  Workspace workspace;
  Output output;

  constexpr int x_dim = 2;
  constexpr int s_dim = 4;
  constexpr int y_dim = 1;
  constexpr int upper_hessian_f_nnz = 3;
  constexpr int jacobian_c_nnz = 2;
  constexpr int jacobian_g_nnz = 4;
  constexpr int upper_jac_g_t_jac_g_nnz = 3;
  constexpr int upper_hessian_f_plus_upper_jac_g_t_jac_g_nnz = 3;
  constexpr int kkt_L_nnz = 3;

  workspace.reserve(settings.lin_sys_formulation, x_dim, s_dim, y_dim,
                    upper_hessian_f_nnz, jacobian_c_nnz, jacobian_g_nnz,
                    upper_jac_g_t_jac_g_nnz,
                    upper_hessian_f_plus_upper_jac_g_t_jac_g_nnz, kkt_L_nnz);

  auto model_callback = [](const ModelCallbackInput &mci,
                           ModelCallbackOutput &mco) -> void {
    mco.f = 0.5 * (4.0 * mci.x[0] * mci.x[0] + 2.0 * mci.x[0] * mci.x[1] +
                   2.0 * mci.x[1] * mci.x[1]) +
            mci.x[0] + mci.x[1];

    mco.gradient_f[0] = 4.0 * mci.x[0] + 1.0 * mci.x[1] + 1.0;
    mco.gradient_f[1] = 1.0 * mci.x[0] + 2.0 * mci.x[1] + 1.0;

    // NOTE: only the upper triangle should be filled.
    mco.upper_hessian_f.rows = x_dim;
    mco.upper_hessian_f.cols = x_dim;
    mco.upper_hessian_f.ind[0] = 0;
    mco.upper_hessian_f.ind[1] = 0;
    mco.upper_hessian_f.ind[2] = 1;
    mco.upper_hessian_f.indptr[0] = 0;
    mco.upper_hessian_f.indptr[1] = 1;
    mco.upper_hessian_f.indptr[2] = 3;
    mco.upper_hessian_f.data[0] = 4.0;
    mco.upper_hessian_f.data[1] = 1.0;
    mco.upper_hessian_f.data[2] = 2.0;
    mco.upper_hessian_f.is_transposed = false;

    mco.c[0] = mci.x[0] + mci.x[1] - 1.0;

    mco.jacobian_c.rows = x_dim;
    mco.jacobian_c.cols = y_dim;
    mco.jacobian_c.ind[0] = 0;
    mco.jacobian_c.ind[1] = 1;
    mco.jacobian_c.indptr[0] = 0;
    mco.jacobian_c.indptr[1] = 2;
    mco.jacobian_c.data[0] = 1.0;
    mco.jacobian_c.data[1] = 1.0;
    mco.jacobian_c.is_transposed = true;

    mco.g[0] = mci.x[0] - 0.7;
    mco.g[1] = -mci.x[0] - 0.0;
    mco.g[2] = mci.x[1] - 0.7;
    mco.g[3] = -mci.x[1] - 0.0;

    mco.jacobian_g.rows = s_dim;
    mco.jacobian_g.cols = x_dim;
    mco.jacobian_g.ind[0] = 0;
    mco.jacobian_g.ind[1] = 1;
    mco.jacobian_g.ind[2] = 2;
    mco.jacobian_g.ind[3] = 3;
    mco.jacobian_g.indptr[0] = 0;
    mco.jacobian_g.indptr[1] = 2;
    mco.jacobian_g.indptr[2] = 4;
    mco.jacobian_g.data[0] = 1.0;
    mco.jacobian_g.data[1] = -1.0;
    mco.jacobian_g.data[2] = 1.0;
    mco.jacobian_g.data[3] = -1.0;
    mco.jacobian_g.is_transposed = false;
  };

  input.model_callback = model_callback;

  for (int i = 0; i < x_dim; ++i) {
    workspace.vars.x[i] = 0.0;
  }

  for (int i = 0; i < s_dim; ++i) {
    workspace.vars.s[i] = 1.0;
    workspace.vars.z[i] = 1.0;
  }

  for (int i = 0; i < y_dim; ++i) {
    workspace.vars.y[i] = 0.0;
  }

  solve(input, settings, workspace, output);

  EXPECT_EQ(output.exit_status, Status::SOLVED);

  EXPECT_NEAR(workspace.vars.x[0], 0.3, 1e-6);
  EXPECT_NEAR(workspace.vars.x[1], 0.7, 1e-6);

  workspace.free();
}

} // namespace sip
