#include "sip.hpp"

#include <algorithm>
#include <cassert>
#include <format>
#include <iostream>
#include <utility>

#include <qdldl.h>

namespace sip {

auto get_max_step_sizes(const int s_dim, const double tau, const double *s,
                        const double *z, const double *ds,
                        const double *dz) -> std::pair<double, double> {
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

auto get_rho(const Workspace &workspace, const double *s, const double *dx,
             const double *ds, const double mu) -> double {
  // D(merit_function; dx, ds) = D(f; dx) - mu (ds / s) - rho * ||c(x)|| - rho *
  // ||g(x) + s || rho > (D(f; dx) + k) / (|| (c(x) || + || g(x) + s) || iff
  // D(merit_function; dx) < -k.
  const auto &mco = workspace.model_callback_output;
  const int x_dim = mco.hessian_f.rows;
  const int s_dim = mco.jacobian_g_transpose.cols;
  const int y_dim = mco.jacobian_c_transpose.cols;
  const double f_slope = dot(mco.gradient_f, dx, x_dim);
  const double barrier_slope = -mu * x_dot_y_inverse(ds, s, s_dim);
  const double obj_slope = f_slope + barrier_slope;
  const double d = norm(mco.c, y_dim) +
                   norm(workspace.miscellaneous_workspace.g_plus_s, s_dim);
  const double k = std::max(d, 2.0 * std::fabs(obj_slope));
  return std::min((obj_slope + k) / d, 1e9);
}

auto merit_function(const Workspace &workspace, const double *s,
                    const double mu, const double rho) -> double {
  const auto &mco = workspace.model_callback_output;
  const int s_dim = mco.jacobian_g_transpose.cols;
  const int y_dim = mco.jacobian_c_transpose.cols;
  return mco.f - mu * sum_of_logs(s, s_dim) + rho * norm(mco.c, y_dim) +
         rho * norm(workspace.miscellaneous_workspace.g_plus_s, s_dim);
}

auto merit_function_slope(const Workspace &workspace, const double *s,
                          const double *dx, const double *ds, const double mu,
                          const double rho) {
  // TODO(joao): eventually remove repeated computation across:
  // 1. merit_function
  // 2. merit_function_slope
  // 3. get_rho
  // 4. build_nrhs (s inversion)
  const auto &mco = workspace.model_callback_output;
  const int x_dim = mco.hessian_f.rows;
  const int s_dim = mco.jacobian_g_transpose.cols;
  const int y_dim = mco.jacobian_c_transpose.cols;
  const double f_slope = dot(mco.gradient_f, dx, x_dim);
  const double barrier_slope = -mu * x_dot_y_inverse(ds, s, s_dim);
  const double obj_slope = f_slope + barrier_slope;
  return obj_slope - rho * norm(mco.c, y_dim) -
         rho * norm(workspace.miscellaneous_workspace.g_plus_s, s_dim);
}

auto build_lhs(const Settings &settings, Workspace &workspace) -> void {
  // Builds the lower triangle of the following matrix in CSC format:
  // [ hessian_f       0        jacobian_c_t     jacobian_g_t ]
  // [     0        S^{-1} Z         0               I_s      ]
  // [     0           0      -gamma_y * I_y          0       ]
  // [     0           0             0         -gamma_z * I_z ]
  const auto &model_callback_output = workspace.model_callback_output;

  const double *s = workspace.vars.s;
  const double *z = workspace.vars.z;

  auto &lhs = workspace.kkt_workspace.lhs;

  const int x_dim = model_callback_output.hessian_f.cols;
  const int s_dim = model_callback_output.jacobian_g_transpose.cols;
  const int y_dim = model_callback_output.jacobian_c_transpose.cols;

  int k = 0;

  // Fill hessian_f.
  for (int i = 0; i < x_dim; ++i) {
    lhs.indptr[i] = k;
    for (int j = model_callback_output.hessian_f.indptr[i];
         j < model_callback_output.hessian_f.indptr[i + 1]; ++j) {
      lhs.ind[k] = model_callback_output.hessian_f.ind[j];
      lhs.data[k] = model_callback_output.hessian_f.data[j];
      ++k;
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
    for (int j = model_callback_output.jacobian_c_transpose.indptr[i];
         j < model_callback_output.jacobian_c_transpose.indptr[i + 1]; ++j) {
      lhs.ind[k] = model_callback_output.jacobian_c_transpose.ind[j];
      lhs.data[k] = model_callback_output.jacobian_c_transpose.data[j];
      ++k;
    }
    // Fill -gamma_y * I_y column.
    lhs.ind[k] = x_dim + s_dim + i;
    lhs.data[k] = -settings.gamma_y;
    ++k;
  }

  // Fill jacobian_g, I_s, and -gamma_z * I_z.
  for (int i = 0; i < s_dim; ++i) {
    lhs.indptr[x_dim + s_dim + y_dim + i] = k;
    // Fill jacobian_g column.
    for (int j = model_callback_output.jacobian_g_transpose.indptr[i];
         j < model_callback_output.jacobian_g_transpose.indptr[i + 1]; ++j) {
      lhs.ind[k] = model_callback_output.jacobian_g_transpose.ind[j];
      lhs.data[k] = model_callback_output.jacobian_g_transpose.data[j];
      ++k;
    }
    // Fill I_s column.
    lhs.ind[k] = x_dim + i;
    lhs.data[k] = 1.0;
    ++k;
    // Fill -gamma_z * I_z column.
    lhs.ind[k] = x_dim + s_dim + y_dim + i;
    lhs.data[k] = -settings.gamma_z;
    ++k;
  }

  lhs.indptr[x_dim + y_dim + 2 * s_dim] = k;
}

auto build_nrhs(const double mu, Workspace &workspace) -> void {
  // Builds the following vector:
  // [ gradient_f + jacobian_c_t @ y + jacobian_g_t @ z ]
  // [                     z - mu / s                   ]
  // [                          c                       ]
  // [                        g + s                     ]
  const auto &model_callback_output = workspace.model_callback_output;

  const double *s = workspace.vars.s;
  const double *y = workspace.vars.y;
  const double *z = workspace.vars.z;

  const int x_dim = model_callback_output.hessian_f.cols;
  const int s_dim = model_callback_output.jacobian_g_transpose.cols;
  const int y_dim = model_callback_output.jacobian_c_transpose.cols;

  double *nrhs = workspace.kkt_workspace.negative_rhs;

  std::copy(model_callback_output.gradient_f,
            model_callback_output.gradient_f + x_dim, nrhs);

  add_Ax_to_y(model_callback_output.jacobian_c_transpose, y, nrhs);
  add_Ax_to_y(model_callback_output.jacobian_g_transpose, z, nrhs);

  for (int i = 0; i < s_dim; ++i) {
    nrhs[x_dim + i] = z[i] - mu / s[i];
  }

  for (int i = 0; i < y_dim; ++i) {
    nrhs[x_dim + s_dim + i] = model_callback_output.c[i];
  }

  for (int i = 0; i < s_dim; ++i) {
    nrhs[x_dim + s_dim + y_dim + i] =
        workspace.miscellaneous_workspace.g_plus_s[i];
  }
}

auto compute_search_direction(const Settings &settings, const double mu,
                              Workspace &workspace)
    -> std::pair<double *, double> {
  const auto &model_callback_output = workspace.model_callback_output;
  build_lhs(settings, workspace);
  build_nrhs(mu, workspace);

  const int x_dim = model_callback_output.hessian_f.cols;
  const int s_dim = model_callback_output.jacobian_g_transpose.cols;
  const int y_dim = model_callback_output.jacobian_c_transpose.cols;

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

  const int num_pos_D_entries = QDLDL_factor(
      dim, workspace.kkt_workspace.lhs.indptr, workspace.kkt_workspace.lhs.ind,
      workspace.kkt_workspace.lhs.data, workspace.qdldl_workspace.Lp,
      workspace.qdldl_workspace.Li, workspace.qdldl_workspace.Lx,
      workspace.qdldl_workspace.D, workspace.qdldl_workspace.Dinv,
      workspace.qdldl_workspace.Lnz, workspace.qdldl_workspace.etree,
      workspace.qdldl_workspace.bwork, workspace.qdldl_workspace.iwork,
      workspace.qdldl_workspace.fwork);

  assert(num_pos_D_entries >= 0);

  for (int i = 0; i < dim; i++) {
    workspace.qdldl_workspace.x[i] = -workspace.kkt_workspace.negative_rhs[i];
  }

  QDLDL_solve(dim, workspace.qdldl_workspace.Lp, workspace.qdldl_workspace.Li,
              workspace.qdldl_workspace.Lx, workspace.qdldl_workspace.Dinv,
              workspace.qdldl_workspace.x);

  return {workspace.qdldl_workspace.x, kkt_error};
}

auto solve(const Input &input, const Settings &settings, Workspace &workspace,
           Output &output) -> void {
  {
    ModelCallbackInput mci{
        .x = workspace.vars.x,
    };

    input.model_callback(mci, workspace.model_callback_output);
  }

  const int x_dim = workspace.model_callback_output.hessian_f.cols;
  const int s_dim = workspace.model_callback_output.jacobian_g_transpose.cols;
  const int y_dim = workspace.model_callback_output.jacobian_c_transpose.cols;

  for (int i = 0; i < s_dim; ++i) {
    assert(workspace.vars.s[i] > 0.0 && workspace.vars.z[i] > 0.0);
  }

  add(workspace.model_callback_output.g, workspace.vars.s, s_dim,
      workspace.miscellaneous_workspace.g_plus_s);

  if (settings.print_logs) {
    std::cout << std::format(
                     // clang-format off
                     "{:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10}",
                     // clang-format on
                     "iteration", "alpha", "merit", "f", "|c|", "|g+s|",
                     "m_slope", "alpha_s_m", "alpha_z_m", "|dx|", "|ds|",
                     "|dy|", "|dz|", "mu")
              << std::endl;
  }

  for (int iteration = 0; iteration < settings.max_iterations; ++iteration) {
    const double mu = adaptive_mu(s_dim, workspace.vars.s, workspace.vars.z);

    const auto [dxsyz, kkt_error] =
        compute_search_direction(settings, mu, workspace);

    if (kkt_error < settings.max_kkt_violation) {
      output.exit_status = Status::SOLVED;
      return;
    }

    const double *dx = dxsyz;
    const double *ds = dxsyz + x_dim;
    const double *dy = dxsyz + x_dim + s_dim;
    const double *dz = dxsyz + x_dim + s_dim + y_dim;

    const double tau = std::max(settings.tau_min, mu > 0.0 ? 1.0 - mu : 0.0);

    const auto [alpha_s_max, alpha_z_max] = get_max_step_sizes(
        s_dim, tau, workspace.vars.s, workspace.vars.z, ds, dz);

    const double rho = get_rho(workspace, workspace.vars.s, dx, ds, mu);

    const double merit = merit_function(workspace, workspace.vars.s, mu, rho);

    const double merit_slope =
        merit_function_slope(workspace, workspace.vars.s, dx, ds, mu, rho);

    bool ls_succeeded = false;
    double alpha = alpha_s_max;
    for (; alpha > settings.line_search_min_step_size;
         alpha *= settings.line_search_factor) {

      for (int i = 0; i < x_dim; ++i) {
        workspace.vars.next_x[i] = workspace.vars.x[i] + alpha * dx[i];
      }

      for (int i = 0; i < s_dim; ++i) {
        workspace.vars.next_s[i] = workspace.vars.s[i] + alpha * ds[i];
      }

      ModelCallbackInput mci{
          .x = workspace.vars.next_x,
      };
      input.model_callback(mci, workspace.model_callback_output);

      add(workspace.model_callback_output.g, workspace.vars.s, s_dim,
          workspace.miscellaneous_workspace.g_plus_s);

      // TODO(joao): cache (parts of) this into the next cycle!
      const double new_merit =
          merit_function(workspace, workspace.vars.next_s, mu, rho);

      if (new_merit - merit < settings.armijo_factor * merit_slope * alpha) {
        ls_succeeded = true;
        break;
      }
    }

    std::swap(workspace.vars.x, workspace.vars.next_x);
    std::swap(workspace.vars.s, workspace.vars.next_s);

    for (int i = 0; i < y_dim; ++i) {
      workspace.vars.y[i] += alpha_z_max * dy[i];
    }

    for (int i = 0; i < s_dim; ++i) {
      workspace.vars.z[i] += alpha_z_max * dz[i];
    }

    if (settings.print_logs) {
      std::cout << std::format(
                       // clang-format off
                       "{:^+10} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g} {:^+10.4g}",
                       // clang-format on
                       iteration, alpha, merit,
                       workspace.model_callback_output.f,
                       norm(workspace.model_callback_output.c, y_dim),
                       norm(workspace.miscellaneous_workspace.g_plus_s, s_dim),
                       merit_slope, alpha_s_max, alpha_z_max, norm(dx, x_dim),
                       norm(ds, s_dim), norm(dy, y_dim), norm(dz, s_dim), mu)
                << std::endl;
    }

    if (!ls_succeeded) {
      output.exit_status = Status::LINE_SEARCH_FAILURE;
      return;
    }
  }

  output.exit_status = Status::ITERATION_LIMIT;
}

void ModelCallbackOutput::reserve(int x_dim, int s_dim, int y_dim,
                                  int hessian_f_nnz, int jacobian_c_nnz,
                                  int jacobian_g_nnz) {
  gradient_f = new double[x_dim];
  hessian_f.reserve(x_dim, hessian_f_nnz);
  c = new double[y_dim];
  jacobian_c_transpose.reserve(y_dim, jacobian_c_nnz);
  g = new double[s_dim];
  jacobian_g_transpose.reserve(s_dim, jacobian_g_nnz);
}

void ModelCallbackOutput::free() {
  ::free(gradient_f);
  hessian_f.free();
  ::free(c);
  jacobian_c_transpose.free();
  ::free(g);
  jacobian_g_transpose.free();
}

void QDLDLWorkspace::reserve(int kkt_dim, int kkt_L_nnz) {
  etree = new int[kkt_dim];
  Lnz = new int[kkt_dim];
  iwork = new int[3 * kkt_dim];
  bwork = new unsigned char[kkt_dim];
  fwork = new double[kkt_dim];
  Lp = new int[kkt_L_nnz];
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

void VariablesWorkspace::reserve(int x_dim, int s_dim, int y_dim) {
  x = new double[x_dim];
  s = new double[s_dim];
  y = new double[y_dim];
  z = new double[s_dim];
  next_x = new double[x_dim];
  next_s = new double[s_dim];
}

void VariablesWorkspace::free() {
  ::free(x);
  ::free(s);
  ::free(y);
  ::free(z);
  ::free(next_x);
  ::free(next_s);
}

void MiscellaneousWorkspace::reserve(int s_dim) {
  g_plus_s = new double[s_dim];
}

void MiscellaneousWorkspace::free() { ::free(g_plus_s); }

void KKTWorkspace::reserve(int kkt_dim, int kkt_nnz) {
  lhs.reserve(kkt_dim, kkt_nnz);
  negative_rhs = new double[kkt_dim];
}

void KKTWorkspace::free() {
  lhs.free();
  ::free(negative_rhs);
}

void Workspace::reserve(int x_dim, int s_dim, int y_dim, int hessian_f_nnz,
                        int jacobian_c_nnz, int jacobian_g_nnz, int kkt_L_nnz) {
  const int kkt_dim = x_dim * 2 * s_dim * y_dim;
  const int kkt_nnz =
      hessian_f_nnz + jacobian_c_nnz + jacobian_g_nnz + 3 * s_dim + y_dim;

  vars.reserve(x_dim, s_dim, y_dim);
  kkt_workspace.reserve(kkt_dim, kkt_nnz);
  qdldl_workspace.reserve(kkt_dim, kkt_L_nnz);
  model_callback_output.reserve(x_dim, s_dim, y_dim, hessian_f_nnz,
                                jacobian_c_nnz, jacobian_g_nnz);
  miscellaneous_workspace.reserve(s_dim);
}

void Workspace::free() {
  vars.free();
  kkt_workspace.free();
  qdldl_workspace.free();
  model_callback_output.free();
  miscellaneous_workspace.free();
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
