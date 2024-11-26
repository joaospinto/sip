#include "sip.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <format>
#include <iostream>
#include <limits>
#include <utility>

#include <qdldl.h>

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
  // ||g(x) + s || rho > (D(f; dx) + k) / (|| (c(x) || + || g(x) + s) || iff
  // D(merit_function; dx) < -k.
  const auto &mco = workspace.model_callback_output;
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
  const auto &mco = workspace.model_callback_output;
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
  // 4. build_nrhs_4x4 (s inversion)
  const auto &mco = workspace.model_callback_output;
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

auto build_lhs_4x4(const Settings &settings, Workspace &workspace) -> void {
  // Builds the following matrix in CSC format:
  // [ upper_hessian_f       0        jacobian_c_t          jacobian_g_t    ]
  // [        0          S^{-1} Z          0                    I_s         ]
  // [        0              0      -gamma_y * I_y               0          ]
  // [        0              0             0         -(gamma_z + 1/p) * I_z ]
  const auto &mco = workspace.model_callback_output;
  const int x_dim = mco.upper_hessian_f.rows;
  const int s_dim = get_s_dim(mco.jacobian_g);
  const int y_dim = get_y_dim(mco.jacobian_c);

  const double *s = workspace.vars.s;
  const double *z = workspace.vars.z;

  auto &lhs = workspace.kkt_workspace.lhs;

  lhs.rows = x_dim + 2 * s_dim + y_dim;
  lhs.cols = lhs.rows;

  int k = 0;

  // Fill upper_hessian_f.
  for (int i = 0; i < x_dim; ++i) {
    lhs.indptr[i] = k;
    for (int j = mco.upper_hessian_f.indptr[i];
         j < mco.upper_hessian_f.indptr[i + 1]; ++j) {
      if (mco.upper_hessian_f.ind[j] <= i) {
        lhs.ind[k] = mco.upper_hessian_f.ind[j];
        lhs.data[k] = mco.upper_hessian_f.data[j];
        ++k;
      }
    }
  }

  // Fill S^{-1} Z.
  for (int i = 0; i < s_dim; ++i) {
    lhs.indptr[x_dim + i] = k;
    lhs.ind[k] = x_dim + i;
    lhs.data[k] = z[i] / s[i];
    ++k;
  }

  // Fill jacobian_c_t and -gamma_y * I_y.
  for (int i = 0; i < y_dim; ++i) {
    lhs.indptr[x_dim + s_dim + i] = k;
    // Fill jacobian_c_t column.
    // NOTE: mco.jacobian_c.is_transposed == true.
    for (int j = mco.jacobian_c.indptr[i]; j < mco.jacobian_c.indptr[i + 1];
         ++j) {
      lhs.ind[k] = mco.jacobian_c.ind[j];
      lhs.data[k] = mco.jacobian_c.data[j];
      ++k;
    }
    // Fill -gamma_y * I_y column.
    lhs.ind[k] = x_dim + s_dim + i;
    lhs.data[k] = -settings.gamma_y;
    ++k;
  }

  // Fill jacobian_g, I_s, and -(gamma_z + 1/p) * I_z.
  for (int i = 0; i < s_dim; ++i) {
    lhs.indptr[x_dim + s_dim + y_dim + i] = k;
    // Fill jacobian_g column.
    // NOTE: mco.jacobian_g.is_transposed == true.
    for (int j = mco.jacobian_g.indptr[i]; j < mco.jacobian_g.indptr[i + 1];
         ++j) {
      lhs.ind[k] = mco.jacobian_g.ind[j];
      lhs.data[k] = mco.jacobian_g.data[j];
      ++k;
    }
    // Fill I_s column.
    lhs.ind[k] = x_dim + i;
    lhs.data[k] = 1.0;
    ++k;
    // Fill -(gamma_z + 1 / p) * I_z column.
    lhs.ind[k] = x_dim + s_dim + y_dim + i;
    lhs.data[k] =
        -settings.gamma_z - (settings.enable_elastics
                                 ? 1.0 / settings.elastic_var_cost_coeff
                                 : 0.0);
    ++k;
  }

  lhs.indptr[x_dim + y_dim + 2 * s_dim] = k;
}

auto build_nrhs_4x4(const Settings &settings, const double mu,
                    Workspace &workspace) -> void {
  // Builds the following vector:
  // [ gradient_f + jacobian_c_t @ y + jacobian_g_t @ z ]
  // [                     z - mu / s                   ]
  // [                          c                       ]
  // [                   g + s - z / p                  ]
  const auto &mco = workspace.model_callback_output;

  const double *s = workspace.vars.s;
  const double *y = workspace.vars.y;
  const double *z = workspace.vars.z;

  const int x_dim = mco.upper_hessian_f.cols;
  const int s_dim = get_s_dim(mco.jacobian_g);
  const int y_dim = get_y_dim(mco.jacobian_c);

  double *nrhs = workspace.kkt_workspace.negative_rhs;

  std::copy(mco.gradient_f, mco.gradient_f + x_dim, nrhs);

  add_ATx_to_y(mco.jacobian_c, y, nrhs);
  add_ATx_to_y(mco.jacobian_g, z, nrhs);

  for (int i = 0; i < s_dim; ++i) {
    nrhs[x_dim + i] = z[i] - mu / s[i];
  }

  for (int i = 0; i < y_dim; ++i) {
    nrhs[x_dim + s_dim + i] = mco.c[i];
  }

  for (int i = 0; i < s_dim; ++i) {
    nrhs[x_dim + s_dim + y_dim + i] =
        workspace.miscellaneous_workspace.g_plus_s[i] -
        (settings.enable_elastics ? z[i] / settings.elastic_var_cost_coeff
                                  : 0.0);
  }
}

auto compute_search_direction_4x4(const Settings &settings, const double mu,
                                  Workspace &workspace)
    -> std::tuple<double *, double *, double *, double *, double *, double,
                  double> {
  build_lhs_4x4(settings, workspace);
  build_nrhs_4x4(settings, mu, workspace);

  const auto &model_callback_output = workspace.model_callback_output;
  const int x_dim = model_callback_output.upper_hessian_f.cols;
  const int s_dim = get_s_dim(model_callback_output.jacobian_g);
  const int y_dim = get_y_dim(model_callback_output.jacobian_c);

  double kkt_error = 0.0;
  for (int i = 0; i < x_dim; ++i) {
    kkt_error =
        std::max(kkt_error, std::fabs(workspace.kkt_workspace.negative_rhs[i]));
  }
  for (int i = 0; i < y_dim; ++i) {
    kkt_error = std::max(
        kkt_error,
        std::fabs(workspace.kkt_workspace.negative_rhs[x_dim + s_dim + i]));
  }
  for (int i = 0; i < s_dim; ++i) {
    kkt_error = std::max(
        kkt_error,
        std::fabs(
            workspace.kkt_workspace.negative_rhs[x_dim + s_dim + y_dim + i]));
  }

  const int dim = x_dim + y_dim + 2 * s_dim;

  const int sumLnz = QDLDL_etree(
      workspace.kkt_workspace.lhs.rows, workspace.kkt_workspace.lhs.indptr,
      workspace.kkt_workspace.lhs.ind, workspace.qdldl_workspace.iwork,
      workspace.qdldl_workspace.Lnz, workspace.qdldl_workspace.etree);

  assert(sumLnz >= 0);

  const int num_pos_D_entries = QDLDL_factor(
      dim, workspace.kkt_workspace.lhs.indptr, workspace.kkt_workspace.lhs.ind,
      workspace.kkt_workspace.lhs.data, workspace.qdldl_workspace.Lp,
      workspace.qdldl_workspace.Li, workspace.qdldl_workspace.Lx,
      workspace.qdldl_workspace.D, workspace.qdldl_workspace.Dinv,
      workspace.qdldl_workspace.Lnz, workspace.qdldl_workspace.etree,
      workspace.qdldl_workspace.bwork, workspace.qdldl_workspace.iwork,
      workspace.qdldl_workspace.fwork);

  assert(num_pos_D_entries >= 0);

  for (int i = 0; i < dim; ++i) {
    workspace.qdldl_workspace.x[i] = -workspace.kkt_workspace.negative_rhs[i];
  }

  QDLDL_solve(dim, workspace.qdldl_workspace.Lp, workspace.qdldl_workspace.Li,
              workspace.qdldl_workspace.Lx, workspace.qdldl_workspace.Dinv,
              workspace.qdldl_workspace.x);

  for (int i = 0; i < dim; ++i) {
    workspace.miscellaneous_workspace.lin_sys_residual[i] =
        workspace.kkt_workspace.negative_rhs[i];
  }

  double lin_sys_error = 0.0;

  add_Ax_to_y_where_A_upper_symmetric(
      workspace.kkt_workspace.lhs, workspace.qdldl_workspace.x,
      workspace.miscellaneous_workspace.lin_sys_residual);

  for (int i = 0; i < dim; ++i) {
    lin_sys_error = std::max(
        lin_sys_error,
        std::fabs(workspace.miscellaneous_workspace.lin_sys_residual[i]));
  }

  double *dx = workspace.qdldl_workspace.x;
  double *ds = dx + x_dim;
  double *dy = ds + s_dim;
  double *dz = dy + y_dim;
  double *de = workspace.vars.de;

  // de = (-dz - (pe + z)) / p
  if (settings.enable_elastics) {
    for (int i = 0; i < s_dim; ++i) {
      const double p = settings.elastic_var_cost_coeff;
      de[i] = (-dz[i] - p * workspace.vars.e[i] - workspace.vars.z[i]) / p;
    }
  }

  return {dx, ds, dy, dz, de, kkt_error, lin_sys_error};
}

auto build_lhs_2x2(const Settings &settings, Workspace &workspace) -> void {
  // Builds the following matrix in CSC format:
  // [ upper_hessian_f + jacobian_g_t @ sigma @ jacobian_g       jacobian_c_t  ]
  // [                          0                               -gamma_y * I_y ]
  // Above, sigma = np.diag(z / (s + (gamma_z + 1/p) * z)).
  const auto &mco = workspace.model_callback_output;
  const int x_dim = mco.upper_hessian_f.rows;
  const int s_dim = get_s_dim(mco.jacobian_g);
  const int y_dim = get_y_dim(mco.jacobian_c);

  const double *s = workspace.vars.s;
  const double *z = workspace.vars.z;

  SparseMatrix &lhs = workspace.kkt_workspace.lhs;

  double *sigma = workspace.miscellaneous_workspace.sigma;

  for (int i = 0; i < s_dim; ++i) {
    sigma[i] =
        z[i] /
        (s[i] + (settings.gamma_z + (settings.enable_elastics
                                         ? 1.0 / settings.elastic_var_cost_coeff
                                         : 0.0)) *
                    z[i]);
  }

  XT_D_X(mco.jacobian_g, sigma,
         workspace.miscellaneous_workspace.jac_g_t_sigma_jac_g);

  add(mco.upper_hessian_f,
      workspace.miscellaneous_workspace.jac_g_t_sigma_jac_g, lhs);

  lhs.rows += y_dim;
  lhs.cols += y_dim;

  // Fill jacobian_c_t and -gamma_y * I_y.
  int k = lhs.indptr[x_dim];
  for (int i = 0; i < y_dim; ++i) {
    // Fill jacobian_c_t column.
    // NOTE: mco.jacobian_c.is_transposed == true.
    for (int j = mco.jacobian_c.indptr[i]; j < mco.jacobian_c.indptr[i + 1];
         ++j) {
      lhs.ind[k] = mco.jacobian_c.ind[j];
      lhs.data[k] = mco.jacobian_c.data[j];
      ++k;
    }
    // Fill -gamma_y * I_y column.
    lhs.ind[k] = x_dim + i;
    lhs.data[k] = -settings.gamma_y;
    ++k;
    lhs.indptr[x_dim + i + 1] = k;
  }
}

auto build_nrhs_2x2(const Settings &settings, const double mu,
                    Workspace &workspace) -> void {
  // Builds the following vector:
  // [ gradient_f + jacobian_c_t @ y + jacobian_g_t @ z
  //         + G.T @ sigma @ (g(x) + (mu / z) - z / p)  ]
  // [                          c                       ]
  const auto &mco = workspace.model_callback_output;

  const double *y = workspace.vars.y;
  const double *z = workspace.vars.z;

  const int x_dim = mco.upper_hessian_f.cols;
  const int s_dim = get_s_dim(mco.jacobian_g);
  const int y_dim = get_y_dim(mco.jacobian_c);

  std::copy(mco.gradient_f, mco.gradient_f + x_dim,
            workspace.miscellaneous_workspace.grad_x_lagrangian);

  add_ATx_to_y(mco.jacobian_c, y,
               workspace.miscellaneous_workspace.grad_x_lagrangian);

  add_ATx_to_y(mco.jacobian_g, z,
               workspace.miscellaneous_workspace.grad_x_lagrangian);

  double *nrhs = workspace.kkt_workspace.negative_rhs;

  std::copy(workspace.miscellaneous_workspace.grad_x_lagrangian,
            workspace.miscellaneous_workspace.grad_x_lagrangian + x_dim, nrhs);

  double *sigma = workspace.miscellaneous_workspace.sigma;

  for (int i = 0; i < s_dim; ++i) {
    workspace.miscellaneous_workspace
        .sigma_times_g_plus_mu_over_z_minus_z_over_p[i] =
        sigma[i] *
        (mco.g[i] + mu / z[i] -
         (settings.enable_elastics ? z[i] / settings.elastic_var_cost_coeff
                                   : 0.0));
  }

  add_ATx_to_y(mco.jacobian_g,
               workspace.miscellaneous_workspace
                   .sigma_times_g_plus_mu_over_z_minus_z_over_p,
               nrhs);

  for (int i = 0; i < y_dim; ++i) {
    nrhs[x_dim + i] = mco.c[i];
  }
}

auto compute_search_direction_2x2(const Settings &settings, const double mu,
                                  Workspace &workspace)
    -> std::tuple<double *, double *, double *, double *, double *, double,
                  double> {
  build_lhs_2x2(settings, workspace);
  build_nrhs_2x2(settings, mu, workspace);

  const auto &model_callback_output = workspace.model_callback_output;
  const int x_dim = model_callback_output.upper_hessian_f.cols;
  const int s_dim = get_s_dim(model_callback_output.jacobian_g);
  const int y_dim = get_y_dim(model_callback_output.jacobian_c);

  const int dim = x_dim + y_dim;

  const int sumLnz = QDLDL_etree(
      workspace.kkt_workspace.lhs.rows, workspace.kkt_workspace.lhs.indptr,
      workspace.kkt_workspace.lhs.ind, workspace.qdldl_workspace.iwork,
      workspace.qdldl_workspace.Lnz, workspace.qdldl_workspace.etree);

  assert(sumLnz >= 0);

  const int num_pos_D_entries = QDLDL_factor(
      dim, workspace.kkt_workspace.lhs.indptr, workspace.kkt_workspace.lhs.ind,
      workspace.kkt_workspace.lhs.data, workspace.qdldl_workspace.Lp,
      workspace.qdldl_workspace.Li, workspace.qdldl_workspace.Lx,
      workspace.qdldl_workspace.D, workspace.qdldl_workspace.Dinv,
      workspace.qdldl_workspace.Lnz, workspace.qdldl_workspace.etree,
      workspace.qdldl_workspace.bwork, workspace.qdldl_workspace.iwork,
      workspace.qdldl_workspace.fwork);

  assert(num_pos_D_entries >= 0);

  for (int i = 0; i < dim; ++i) {
    workspace.qdldl_workspace.x[i] = -workspace.kkt_workspace.negative_rhs[i];
  }

  QDLDL_solve(dim, workspace.qdldl_workspace.Lp, workspace.qdldl_workspace.Li,
              workspace.qdldl_workspace.Lx, workspace.qdldl_workspace.Dinv,
              workspace.qdldl_workspace.x);

  double *dx = workspace.qdldl_workspace.x;
  double *dy = dx + x_dim;
  double *ds = workspace.vars.ds;
  double *dz = workspace.vars.dz;
  double *de = workspace.vars.de;

  // dz = sigma @ (g(x) + G @ dx + (mu / z - z / p))
  for (int i = 0; i < s_dim; ++i) {
    dz[i] = workspace.miscellaneous_workspace
                .sigma_times_g_plus_mu_over_z_minus_z_over_p[i];
  }

  add_weighted_Ax_to_y(workspace.model_callback_output.jacobian_g,
                       workspace.miscellaneous_workspace.sigma, dx, dz);

  // ds = z / p -(g(x) + s) + (gamma_z + 1 / p) * dz - G @ dx
  std::copy(workspace.miscellaneous_workspace.g_plus_s,
            workspace.miscellaneous_workspace.g_plus_s + s_dim, ds);

  add_Ax_to_y(workspace.model_callback_output.jacobian_g, dx, ds);

  for (int i = 0; i < s_dim; ++i) {
    ds[i] = -ds[i] + settings.gamma_z * dz[i];
    if (settings.enable_elastics) {
      ds[i] += (workspace.vars.z[i] + dz[i]) / settings.elastic_var_cost_coeff;
    }
  }

  // de = (-dz - (pe + z)) / p
  if (settings.enable_elastics) {
    for (int i = 0; i < s_dim; ++i) {
      const double p = settings.elastic_var_cost_coeff;
      de[i] = (-dz[i] - p * workspace.vars.e[i] - workspace.vars.z[i]) / p;
    }
  }

  double lin_sys_error = 0.0;

  for (int i = 0; i < dim; ++i) {
    workspace.miscellaneous_workspace.lin_sys_residual[i] =
        workspace.kkt_workspace.negative_rhs[i];
  }

  add_Ax_to_y_where_A_upper_symmetric(
      workspace.kkt_workspace.lhs, workspace.qdldl_workspace.x,
      workspace.miscellaneous_workspace.lin_sys_residual);

  for (int i = 0; i < dim; ++i) {
    lin_sys_error = std::max(
        lin_sys_error,
        std::fabs(workspace.miscellaneous_workspace.lin_sys_residual[i]));
  }

  double kkt_error = 0.0;
  for (int i = 0; i < x_dim; ++i) {
    kkt_error = std::max(
        kkt_error,
        std::fabs(workspace.miscellaneous_workspace.grad_x_lagrangian[i]));
  }
  for (int i = 0; i < y_dim; ++i) {
    kkt_error =
        std::max(kkt_error, std::fabs(workspace.model_callback_output.c[i]));
  }
  for (int i = 0; i < s_dim; ++i) {
    kkt_error = std::max(
        kkt_error,
        std::fabs(workspace.miscellaneous_workspace.g_plus_s_plus_e[i]));
  }

  return {dx, ds, dy, dz, de, kkt_error, lin_sys_error};
}

auto compute_search_direction(const Settings &settings, const double mu,
                              Workspace &workspace)
    -> std::tuple<double *, double *, double *, double *, double *, double,
                  double> {

  switch (settings.lin_sys_formulation) {
  case Settings::LinearSystemFormulation::SYMMETRIC_DIRECT_4x4:
    return compute_search_direction_4x4(settings, mu, workspace);
  case Settings::LinearSystemFormulation::SYMMETRIC_INDIRECT_2x2:
    return compute_search_direction_2x2(settings, mu, workspace);
  }
}

auto check_inputs(const ModelCallbackOutput &mco, const Settings &settings) {
  assert(!settings.enable_elastics || settings.elastic_var_cost_coeff > 0.0);
  assert(mco.jacobian_c.is_transposed);
  switch (settings.lin_sys_formulation) {
  case Settings::LinearSystemFormulation::SYMMETRIC_DIRECT_4x4:
    assert(mco.jacobian_g.is_transposed);
    break;
  case Settings::LinearSystemFormulation::SYMMETRIC_INDIRECT_2x2:
    assert(!mco.jacobian_g.is_transposed);
    break;
  }
}

auto solve(const Input &input, const Settings &settings, Workspace &workspace,
           Output &output) -> void {
  {
    ModelCallbackInput mci{
        .x = workspace.vars.x,
    };

    input.model_callback(mci, workspace.model_callback_output);
  }

  check_inputs(workspace.model_callback_output, settings);

  const int x_dim = workspace.model_callback_output.upper_hessian_f.cols;
  const int s_dim = get_s_dim(workspace.model_callback_output.jacobian_g);
  const int y_dim = get_y_dim(workspace.model_callback_output.jacobian_c);

  for (int i = 0; i < s_dim; ++i) {
    assert(workspace.vars.s[i] > 0.0 && workspace.vars.z[i] > 0.0);
  }

  add(workspace.model_callback_output.g, workspace.vars.s, s_dim,
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
    std::cout << std::format(
                     // clang-format off
                     "{:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10}",
                     // clang-format on
                     "iteration", "alpha", "merit", "f", "|c|", "|g+s+e|",
                     "m_slope", "alpha_s_m", "alpha_z_m", "|dx|", "|ds|",
                     "|dy|", "|dz|", "|de|", "mu", "linsys_res")
              << std::endl;
  }

  for (int iteration = 0; iteration < settings.max_iterations; ++iteration) {
    const double mu =
        std::max(adaptive_mu(s_dim, workspace.vars.s, workspace.vars.z),
                 settings.mu_min);

    const auto [dx, ds, dy, dz, de, kkt_error, lin_sys_error] =
        compute_search_direction(settings, mu, workspace);

    if (kkt_error < settings.max_kkt_violation) {
      output.exit_status = Status::SOLVED;
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
      input.model_callback(mci, workspace.model_callback_output);

      add(workspace.model_callback_output.g, workspace.vars.next_s, s_dim,
          workspace.miscellaneous_workspace.g_plus_s);

      if (settings.enable_elastics) {
        add(workspace.miscellaneous_workspace.g_plus_s, workspace.vars.e, s_dim,
            workspace.miscellaneous_workspace.g_plus_s_plus_e);
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
      std::cout << std::format(
                       // clang-format off
                       "{:^+10} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g}",
                       // clang-format on
                       iteration, alpha, new_merit,
                       workspace.model_callback_output.f,
                       norm(workspace.model_callback_output.c, y_dim),
                       norm(workspace.miscellaneous_workspace.g_plus_s_plus_e,
                            s_dim),
                       merit_slope, alpha_s_max, alpha_z_max, norm(dx, x_dim),
                       norm(ds, s_dim), norm(dy, y_dim), norm(dz, s_dim),
                       de_norm, mu, lin_sys_error)
                << std::endl;
    }

    if (settings.enable_line_search_failures && !ls_succeeded) {
      output.exit_status = Status::LINE_SEARCH_FAILURE;
      return;
    }
  }

  output.exit_status = Status::ITERATION_LIMIT;
}

void ModelCallbackOutput::reserve(
    Settings::LinearSystemFormulation lin_sys_formulation, int x_dim, int s_dim,
    int y_dim, int upper_hessian_f_nnz, int jacobian_c_nnz,
    int jacobian_g_nnz) {
  gradient_f = new double[x_dim];
  upper_hessian_f.reserve(x_dim, upper_hessian_f_nnz);
  c = new double[y_dim];
  jacobian_c.reserve(y_dim, jacobian_c_nnz);
  g = new double[s_dim];
  switch (lin_sys_formulation) {
  case Settings::LinearSystemFormulation::SYMMETRIC_DIRECT_4x4:
    jacobian_g.reserve(s_dim, jacobian_g_nnz);
    break;
  case Settings::LinearSystemFormulation::SYMMETRIC_INDIRECT_2x2:
    jacobian_g.reserve(x_dim, jacobian_g_nnz);
    break;
  }
}

void ModelCallbackOutput::free() {
  ::free(gradient_f);
  upper_hessian_f.free();
  ::free(c);
  jacobian_c.free();
  ::free(g);
  jacobian_g.free();
}

auto ModelCallbackOutput::mem_assign(
    Settings::LinearSystemFormulation lin_sys_formulation, int x_dim, int s_dim,
    int y_dim, int upper_hessian_f_nnz, int jacobian_c_nnz, int jacobian_g_nnz,
    unsigned char *mem_ptr) -> int {
  int cum_size = 0;
  gradient_f = reinterpret_cast<decltype(gradient_f)>(mem_ptr + cum_size);
  cum_size += x_dim * sizeof(double);

  cum_size += upper_hessian_f.mem_assign(x_dim, upper_hessian_f_nnz,
                                         mem_ptr + cum_size);

  c = reinterpret_cast<decltype(c)>(mem_ptr + cum_size);
  cum_size += y_dim * sizeof(double);

  cum_size += jacobian_c.mem_assign(y_dim, jacobian_c_nnz, mem_ptr + cum_size);

  g = reinterpret_cast<decltype(g)>(mem_ptr + cum_size);
  cum_size += s_dim * sizeof(double);

  switch (lin_sys_formulation) {
  case Settings::LinearSystemFormulation::SYMMETRIC_DIRECT_4x4:
    cum_size +=
        jacobian_g.mem_assign(s_dim, jacobian_g_nnz, mem_ptr + cum_size);
    break;
  case Settings::LinearSystemFormulation::SYMMETRIC_INDIRECT_2x2:
    cum_size +=
        jacobian_g.mem_assign(x_dim, jacobian_g_nnz, mem_ptr + cum_size);
    break;
  }

  return cum_size;
}

void QDLDLWorkspace::reserve(int kkt_dim, int kkt_L_nnz) {
  etree = new int[kkt_dim];
  Lnz = new int[kkt_dim];
  iwork = new int[3 * kkt_dim];
  bwork = new unsigned char[kkt_dim];
  fwork = new double[kkt_dim];
  Lp = new int[kkt_dim + 1];
  Li = new int[kkt_L_nnz];
  Lx = new double[kkt_L_nnz];
  D = new double[kkt_dim];
  Dinv = new double[kkt_dim];
  x = new double[kkt_dim];
}

void QDLDLWorkspace::free() {
  ::free(etree);
  ::free(Lnz);
  ::free(iwork);
  ::free(bwork);
  ::free(fwork);
  ::free(Lp);
  ::free(Li);
  ::free(Lx);
  ::free(D);
  ::free(Dinv);
  ::free(x);
}

auto QDLDLWorkspace::mem_assign(int kkt_dim, int kkt_L_nnz,
                                unsigned char *mem_ptr) -> int {
  int cum_size = 0;

  etree = reinterpret_cast<decltype(etree)>(mem_ptr + cum_size);
  cum_size += kkt_dim * sizeof(int);

  Lnz = reinterpret_cast<decltype(Lnz)>(mem_ptr + cum_size);
  cum_size += kkt_dim * sizeof(int);

  iwork = reinterpret_cast<decltype(iwork)>(mem_ptr + cum_size);
  cum_size += 3 * kkt_dim * sizeof(int);

  bwork = reinterpret_cast<decltype(bwork)>(mem_ptr + cum_size);
  cum_size += kkt_dim * sizeof(unsigned char);

  fwork = reinterpret_cast<decltype(fwork)>(mem_ptr + cum_size);
  cum_size += kkt_dim * sizeof(double);

  Lp = reinterpret_cast<decltype(Lp)>(mem_ptr + cum_size);
  cum_size += (kkt_dim + 1) * sizeof(int);

  Li = reinterpret_cast<decltype(Li)>(mem_ptr + cum_size);
  cum_size += kkt_L_nnz * sizeof(int);

  Lx = reinterpret_cast<decltype(Lx)>(mem_ptr + cum_size);
  cum_size += kkt_L_nnz * sizeof(double);

  D = reinterpret_cast<decltype(D)>(mem_ptr + cum_size);
  cum_size += kkt_dim * sizeof(double);

  Dinv = reinterpret_cast<decltype(Dinv)>(mem_ptr + cum_size);
  cum_size += kkt_dim * sizeof(double);

  x = reinterpret_cast<decltype(x)>(mem_ptr + cum_size);
  cum_size += kkt_dim * sizeof(double);

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
  ds = new double[s_dim];
  dz = new double[s_dim];
  de = new double[s_dim];
}

void VariablesWorkspace::free() {
  ::free(x);
  ::free(s);
  ::free(y);
  ::free(z);
  ::free(e);
  ::free(next_x);
  ::free(next_s);
  ::free(next_e);
  ::free(ds);
  ::free(dz);
  ::free(de);
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

  ds = reinterpret_cast<decltype(ds)>(mem_ptr + cum_size);
  cum_size += s_dim * sizeof(double);

  dz = reinterpret_cast<decltype(dz)>(mem_ptr + cum_size);
  cum_size += s_dim * sizeof(double);

  de = reinterpret_cast<decltype(de)>(mem_ptr + cum_size);
  cum_size += s_dim * sizeof(double);

  return cum_size;
}

void MiscellaneousWorkspace::reserve(int x_dim, int s_dim, int kkt_dim,
                                     int upper_jac_g_t_jac_g_nnz) {
  g_plus_s = new double[s_dim];
  g_plus_s_plus_e = new double[s_dim];
  lin_sys_residual = new double[kkt_dim];
  grad_x_lagrangian = new double[x_dim];
  sigma = new double[s_dim];
  sigma_times_g_plus_mu_over_z_minus_z_over_p = new double[s_dim];
  jac_g_t_sigma_jac_g.reserve(x_dim, upper_jac_g_t_jac_g_nnz);
}

void MiscellaneousWorkspace::free() {
  ::free(g_plus_s);
  ::free(g_plus_s_plus_e);
  ::free(lin_sys_residual);
  ::free(grad_x_lagrangian);
  ::free(sigma);
  ::free(sigma_times_g_plus_mu_over_z_minus_z_over_p);
  jac_g_t_sigma_jac_g.free();
}

auto MiscellaneousWorkspace::mem_assign(int x_dim, int s_dim, int kkt_dim,
                                        int jac_g_t_jac_g_nnz,
                                        unsigned char *mem_ptr) -> int {
  int cum_size = 0;

  g_plus_s = reinterpret_cast<decltype(g_plus_s)>(mem_ptr + cum_size);
  cum_size += s_dim * sizeof(double);

  g_plus_s_plus_e =
      reinterpret_cast<decltype(g_plus_s_plus_e)>(mem_ptr + cum_size);
  cum_size += s_dim * sizeof(double);

  lin_sys_residual =
      reinterpret_cast<decltype(lin_sys_residual)>(mem_ptr + cum_size);
  cum_size += kkt_dim * sizeof(double);

  grad_x_lagrangian =
      reinterpret_cast<decltype(grad_x_lagrangian)>(mem_ptr + cum_size);
  cum_size += x_dim * sizeof(double);

  sigma = reinterpret_cast<decltype(sigma)>(mem_ptr + cum_size);
  cum_size += s_dim * sizeof(double);

  sigma_times_g_plus_mu_over_z_minus_z_over_p =
      reinterpret_cast<decltype(sigma_times_g_plus_mu_over_z_minus_z_over_p)>(
          mem_ptr + cum_size);
  cum_size += s_dim * sizeof(double);

  cum_size += jac_g_t_sigma_jac_g.mem_assign(x_dim, jac_g_t_jac_g_nnz,
                                             mem_ptr + cum_size);

  return cum_size;
}

void KKTWorkspace::reserve(int kkt_dim, int kkt_nnz) {
  lhs.reserve(kkt_dim, kkt_nnz);
  negative_rhs = new double[kkt_dim];
}

void KKTWorkspace::free() {
  lhs.free();
  ::free(negative_rhs);
}

auto KKTWorkspace::mem_assign(int kkt_dim, int kkt_nnz, unsigned char *mem_ptr)
    -> int {
  int cum_size = 0;

  cum_size += lhs.mem_assign(kkt_dim, kkt_nnz, mem_ptr + cum_size);

  negative_rhs = reinterpret_cast<decltype(negative_rhs)>(mem_ptr + cum_size);
  cum_size += kkt_dim * sizeof(double);

  return cum_size;
}

auto get_kkt_dim(Settings::LinearSystemFormulation lin_sys_formulation,
                 int x_dim, int s_dim, int y_dim) -> int {
  switch (lin_sys_formulation) {
  case Settings::LinearSystemFormulation::SYMMETRIC_DIRECT_4x4:
    return x_dim + 2 * s_dim + y_dim;
  case Settings::LinearSystemFormulation::SYMMETRIC_INDIRECT_2x2:
    return x_dim + y_dim;
  }
};

auto get_kkt_nnz(Settings::LinearSystemFormulation lin_sys_formulation,
                 int upper_hessian_f_nnz, int jacobian_c_nnz,
                 int jacobian_g_nnz,
                 int upper_hessian_f_plus_upper_jac_g_t_jac_g_nnz, int s_dim,
                 int y_dim) {
  switch (lin_sys_formulation) {
  case Settings::LinearSystemFormulation::SYMMETRIC_DIRECT_4x4:
    return upper_hessian_f_nnz + jacobian_c_nnz + jacobian_g_nnz + 3 * s_dim +
           y_dim;
  case Settings::LinearSystemFormulation::SYMMETRIC_INDIRECT_2x2:
    return upper_hessian_f_plus_upper_jac_g_t_jac_g_nnz + jacobian_c_nnz +
           y_dim;
  }
};

void Workspace::reserve(Settings::LinearSystemFormulation lin_sys_formulation,
                        int x_dim, int s_dim, int y_dim,
                        int upper_hessian_f_nnz, int jacobian_c_nnz,
                        int jacobian_g_nnz, int upper_jac_g_t_jac_g_nnz,
                        int upper_hessian_f_plus_upper_jac_g_t_jac_g_nnz,
                        int kkt_L_nnz) {
  const int kkt_dim = get_kkt_dim(lin_sys_formulation, x_dim, s_dim, y_dim);
  const int kkt_nnz = get_kkt_nnz(
      lin_sys_formulation, upper_hessian_f_nnz, jacobian_c_nnz, jacobian_g_nnz,
      upper_hessian_f_plus_upper_jac_g_t_jac_g_nnz, s_dim, y_dim);

  vars.reserve(x_dim, s_dim, y_dim);
  kkt_workspace.reserve(kkt_dim, kkt_nnz);
  qdldl_workspace.reserve(kkt_dim, kkt_L_nnz);
  model_callback_output.reserve(lin_sys_formulation, x_dim, s_dim, y_dim,
                                upper_hessian_f_nnz, jacobian_c_nnz,
                                jacobian_g_nnz);
  miscellaneous_workspace.reserve(x_dim, s_dim, kkt_dim,
                                  upper_jac_g_t_jac_g_nnz);
}

void Workspace::free() {
  vars.free();
  kkt_workspace.free();
  qdldl_workspace.free();
  model_callback_output.free();
  miscellaneous_workspace.free();
}

auto Workspace::mem_assign(
    Settings::LinearSystemFormulation lin_sys_formulation, int x_dim, int s_dim,
    int y_dim, int upper_hessian_f_nnz, int jacobian_c_nnz,
    int jac_g_t_jac_g_nnz, int jacobian_g_nnz,
    int upper_hessian_f_plus_upper_jac_g_t_jac_g_nnz, int kkt_L_nnz,
    unsigned char *mem_ptr) -> int {
  const int kkt_dim = get_kkt_dim(lin_sys_formulation, x_dim, s_dim, y_dim);
  const int kkt_nnz = get_kkt_nnz(
      lin_sys_formulation, upper_hessian_f_nnz, jacobian_c_nnz, jacobian_g_nnz,
      upper_hessian_f_plus_upper_jac_g_t_jac_g_nnz, s_dim, y_dim);

  int cum_size = 0;

  cum_size += vars.mem_assign(x_dim, s_dim, y_dim, mem_ptr + cum_size);
  cum_size += kkt_workspace.mem_assign(kkt_dim, kkt_nnz, mem_ptr + cum_size);
  cum_size +=
      qdldl_workspace.mem_assign(kkt_dim, kkt_L_nnz, mem_ptr + cum_size);
  cum_size += model_callback_output.mem_assign(
      lin_sys_formulation, x_dim, s_dim, y_dim, upper_hessian_f_nnz,
      jacobian_c_nnz, jacobian_g_nnz, mem_ptr + cum_size);
  cum_size += miscellaneous_workspace.mem_assign(
      x_dim, s_dim, kkt_dim, jac_g_t_jac_g_nnz, mem_ptr + cum_size);

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
