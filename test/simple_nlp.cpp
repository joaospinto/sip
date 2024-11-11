#include "sip/sip.hpp"

#include <gtest/gtest.h>

namespace sip {

TEST(SimpleNLP, Problem1) {
  Input input;
  Settings settings{.max_kkt_violation = 1e-12};
  Workspace workspace;
  Output output;

  constexpr int x_dim = 2;
  constexpr int s_dim = 2;
  constexpr int y_dim = 0;
  constexpr int upper_hessian_f_nnz = 3;
  constexpr int jacobian_c_nnz = 0;
  constexpr int jacobian_g_nnz = 4;
  constexpr int kkt_L_nnz = 13;

  workspace.reserve(x_dim, s_dim, y_dim, upper_hessian_f_nnz, jacobian_c_nnz,
                    jacobian_g_nnz, kkt_L_nnz);

  auto model_callback = [](const ModelCallbackInput &mci,
                           ModelCallbackOutput &mco) -> void {
    mco.f = mci.x[1] * (5.0 + mci.x[0]);

    mco.gradient_f[0] = mci.x[1];
    mco.gradient_f[1] = 5.0 + mci.x[0];

    // NOTE: a positive definite Hessian approximation is expected.
    mco.upper_hessian_f.rows = x_dim;
    mco.upper_hessian_f.cols = x_dim;
    mco.upper_hessian_f.nnz = upper_hessian_f_nnz;
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

    // No equality constraints, so we don't set mco.c.

    mco.jacobian_c_transpose.rows = x_dim;
    mco.jacobian_c_transpose.cols = y_dim;
    mco.jacobian_c_transpose.nnz = jacobian_c_nnz;
    mco.jacobian_c_transpose.indptr[0] = 0;

    mco.g[0] = 5.0 - mci.x[0] * mci.x[1];
    mco.g[1] = mci.x[0] * mci.x[0] + mci.x[1] * mci.x[1] - 20.0;

    mco.jacobian_g_transpose.rows = x_dim;
    mco.jacobian_g_transpose.cols = s_dim;
    mco.jacobian_g_transpose.nnz = jacobian_g_nnz;
    mco.jacobian_g_transpose.ind[0] = 0;
    mco.jacobian_g_transpose.ind[1] = 1;
    mco.jacobian_g_transpose.ind[2] = 0;
    mco.jacobian_g_transpose.ind[3] = 1;
    mco.jacobian_g_transpose.indptr[0] = 0;
    mco.jacobian_g_transpose.indptr[1] = 2;
    mco.jacobian_g_transpose.indptr[2] = 4;
    mco.jacobian_g_transpose.data[0] = -mci.x[1];
    mco.jacobian_g_transpose.data[1] = -mci.x[0];
    mco.jacobian_g_transpose.data[2] = 2 * mci.x[0];
    mco.jacobian_g_transpose.data[3] = 2 * mci.x[1];
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

  EXPECT_NEAR(workspace.vars.x[0], -1.15747396, 1e-6);
  EXPECT_NEAR(workspace.vars.x[1], -4.31975162, 1e-6);

  workspace.free();
}

} // namespace sip
