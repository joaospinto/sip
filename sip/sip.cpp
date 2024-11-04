#include "sip.hpp"

#include <algorithm>
#include <utility>

// TODO(joao): are we getting any value from Eigen?
#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <qdldl.h>

namespace sip {

auto eigen_view(const double *x, const int dim) -> Eigen::Ref<Eigen::VectorXd> {
  return Eigen::Ref<Eigen::VectorXd>(x, dim);
}

auto eigen_view(const SparseMatrix &M)
    -> Eigen::Map<Eigen::SparseMatrix<double>> {
  return Eigen::Map<Eigen::SparseMatrix<double>>(M.rows, M.cols, M.nnz,
                                                 M.indptr, M.ind, M.data);
}

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

auto barrier_augmented_lagrangian(
    const ModelCallbackOutput &model_callback_output,
    const Eigen::Ref<const Eigen::VectorXd> s,
    const Eigen::Ref<const Eigen::VectorXd> y,
    const Eigen::Ref<const Eigen::VectorXd> z, const double mu) -> double {
  return model_callback_output.f + y.dot(model_callback_output.c) +
         z.dot(model_callback_output.g + s) - mu * s.array().log().sum();
}

auto adaptive_mu(const Eigen::Ref<const Eigen::VectorXd> s,
                 const Eigen::Ref<const Eigen::VectorXd> z) -> double {
  // Uses the LOQO rule mentioned in Nocedal & Wright.
  const int m = s.size();
  const double dot = s.dot(z);
  const double zeta = s.colwiseProduct(z).min() * m / dot;
  const auto cube = [](const double k) { return k * k * k; };
  const double sigma = 0.1 * cube(std::min(0.5 * (1.0 - zeta) / zeta, 2.0));
  return sigma * dot / m;
}

auto get_rho(const ModelCallbackOutput &model_callback_output,
             const Eigen::Ref<const Eigen::VectorXd> x,
             const Eigen::Ref<const Eigen::VectorXd> s,
             const Eigen::Ref<const Eigen::VectorXd> dx,
             const Eigen::Ref<const Eigen::VectorXd> ds,
             const double mu) -> double {
  // D(merit_function; dx, ds) = D(f; dx) - mu (ds / s) - rho * ||c(x)|| - rho *
  // ||g(x) + s || rho > (D(f; dx) + k) / (|| (c(x) || + || g(x) + s) || iff
  // D(merit_function; dx) < -k.
  const double f_slope =
      eigen_view(model_callback_output.gradient_f, x.size()).dot(dx);
  const double barrier_slope = -mu * s.cwiseInverse().dot(ds);
  const double obj_slope = f_slope + barrier_slope;
  const double d =
      model_callback_output.c.norm() + (model_callback_output.g + s).norm();
  const double k = std::max(d, 2.0 * std::fabs(obj_slope));
  return std::min((obj_slope + k) / d, 1e9);
}

auto merit_function(const ModelCallbackOutput &model_callback_output,
                    const Eigen::Ref<const Eigen::VectorXd> x,
                    const Eigen::Ref<const Eigen::VectorXd> s, const double mu,
                    const double rho) -> double {
  return model_callback_output.f - mu * s.array().log().sum() +
         rho * model_callback_output.c.norm() +
         rho * (model_callback_output.g + s).norm();
}

auto merit_function_slope(const ModelCallbackOutput &model_callback_output,
                          const Eigen::Ref<const Eigen::VectorXd> x,
                          const Eigen::Ref<const Eigen::VectorXd> s,
                          const Eigen::Ref<const Eigen::VectorXd> dx,
                          const Eigen::Ref<const Eigen::VectorXd> ds,
                          const double mu, const double rho) {
  // TODO(joao): eventually remove repeated computation across:
  // 1. merit_function
  // 2. merit_function_slope
  // 3. get_rho
  // 4. build_rhs (s inversion)
  return eigen_view(model_callback_output.gradient_f, x.size()).dot(dx) -
         mu * s.cwiseInverse().dot(ds) - rho * model_callback_output.c.norm() -
         rho * (model_callback_output.g + s).norm();
}

auto build_lhs(const ModelCallbackOutput &model_callback_output,
               const Settings &settings, double *s, double *z, const double mu,
               Workspace &workspace) -> void {
  // Builds the lower triangle of the following matrix in CSC format:
  // [ hessian_f       0             0                0       ]
  // [     0        S^{-1} Z         0                0       ]
  // [ jacobian_c   0         -gamma_y * I_y          0       ]
  // [ jacobian_g   I_s              0         -gamma_z * I_z ]
  auto &lhs = workspace.kkt_workspace.lhs;

  const int x_dim = model_callback_output.hessian_f.cols;
  const int s_dim = model_callback_output.jacobian_g.rows;
  const int y_dim = model_callback_output.jacobian_c.rows;

  int k = 0;

  // Fill hessian_f, jacobian_c, and jacobian_g.
  for (int i = 0; i < x_dim; ++i) {
    lhs.indptr[i] = k;
    // Fill hessian_f column.
    for (int j = model_callback_output.hessian_f.indptr[i];
         j < model_callback_output.hessian_f.indptr[i + 1]; ++j) {
      lhs.ind[k] = model_callback_output.hessian_f.ind[j];
      lhs.data[k] = model_callback_output.hessian_f.data[j];
      ++k;
    }
    // Fill jacobian_c column.
    for (int j = model_callback_output.jacobian_c.indptr[i];
         j < model_callback_output.jacobian_c.indptr[i + 1]; ++j) {
      lhs.ind[k] = model_callback_output.jacobian_c.ind[j] + x_dim + s_dim;
      lhs.data[k] = model_callback_output.jacobian_c.data[j];
      ++k;
    }
    // Fill jacobian_g column.
    for (int j = model_callback_output.jacobian_g.indptr[i];
         j < model_callback_output.jacobian_g.indptr[i + 1]; ++j) {
      lhs.ind[k] =
          model_callback_output.jacobian_g.ind[j] + x_dim + s_dim + y_dim;
      lhs.data[k] = model_callback_output.jacobian_g.data[j];
      ++k;
    }
  }

  // Fill S^{-1} Z and I_s.
  for (int i = 0; i < s_dim; ++i) {
    lhs.indptr[x_dim + i] = k;
    lhs.ind[k] = x_dim + i;
    lhs.data[k] = z[i] / s[i];
    ++k;
    lhs.ind[k] = x_dim + s_dim + y_dim + i;
    lhs.data[k] = 1.0;
    ++k;
  }

  // Fill -gamma_y * I_y.
  for (int i = 0; i < y_dim; ++i) {
    lhs.indptr[x_dim + s_dim + i] = k;
    lhs.ind[k] = x_dim + s_dim + i;
    lhs.data[k] = -settings.gamma_y;
    ++k;
  }

  // Fill -gamma_z * I_z.
  for (int i = 0; i < z_dim; ++i) {
    lhs.indptr[x_dim + s_dim + y_dim + i] = k;
    lhs.ind[k] = x_dim + s_dim + y_dim + i;
    lhs.data[k] = -settings.gamma_z;
    ++k;
  }

  lhs.indptr[x_dim + y_dim + 2 * s_dim] = k;
}

auto build_rhs(const ModelCallbackOutput &model_callback_output,
               const double *s, const double *y, const double *z,
               const double mu, Workspace &workspace) -> void {
  // Builds the following vector:
  // [ hessian_f  ]
  // [      0     ]
  // [ jacobian_c ]
  // [ jacobian_g ]
  const int x_dim = model_callback_output.hessian_f.cols;
  const int s_dim = model_callback_output.jacobian_g.rows;
  const int y_dim = model_callback_output.jacobian_c.rows;
  double *nrhs = workspace.kkt_workspace.negative_rhs;
  std::copy(model_callback_output.gradient_f,
            model_callback_output.gradient_f + x_dim, nrhs);
  add_ATx_to_y(model_callback_output.jacobian_c, y, nrhs);
  add_ATx_to_y(model_callback_output.jacobian_g, z, nrhs);
  for (int i = 0; i < dim_s; ++i) {
    nrhs[x_dim + i] = z[i] - mu / s[i];
  }
  for (int i = 0; i < y_dim; ++i) {
    nrhs[x_dim + s_dim + i] = model_callback_output.c[i];
  }
  for (int i = 0; i < s_dim; ++i) {
    nrhs[x_dim + s_dim + y_dim + i] = model_callback_output.g[i] + s[i];
  }
}

auto compute_search_direction(const ModelCallbackOutput &model_callback_output,
                              const Settings &settings, double *s, double *y,
                              double *z, const double mu, Workspace &workspace)
    -> std::pair<double *, double> {
  build_lhs(model_callback_output, settings, s, z, mu, workspace);
  build_rhs(model_callback_output, s, y, z, mu, workspace);

  const int x_dim = model_callback_output.hessian_f.cols;
  const int s_dim = model_callback_output.jacobian_g.rows;
  const int y_dim = model_callback_output.jacobian_c.rows;

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
  for (int i = 0; i < z_dim; ++i) {
    kkt_error = std::max(
        kkt_error,
        std::fabs(
            workspace.kkt_workspace.negative_rhs[x_dim + s_dim + y_dim + i]));
  }

  const int dim = x_dim + y_dim + 2 * s_dim;

  QDLDL_factor(dim, workspace.kkt_workspace.lhs.indptr,
               workspace.kkt_workspace.lhs.ind,
               workspace.kkt_workspace.lhs.data, workspace.qdldl_workspace.Lp,
               workspace.qdldl_workspace.Li, workspace.qdldl_workspace.Lx,
               workspace.qdldl_workspace.D, workspace.qdldl_workspace.Dinv,
               workspace.qdldl_workspace.Lnz, workspace.qdldl_workspace.etree,
               workspace.qdldl_workspace.bwork, workspace.qdldl_workspace.iwork,
               workspace.qdldl_workspace.fwork);

  for (int i = 0; i < dim; i++) {
    workspace.qdldl_workspace.x[i] = -workspace.kkt_workspace.negative_rhs[i];
  }

  QDLDL_solve(workspace.qdldl_workspace.Ln, workspace.qdldl_workspace.Lp,
              workspace.qdldl_workspace.Li, workspace.qdldl_workspace.Lx,
              workspace.qdldl_workspace.Dinv, workspace.qdldl_workspace.x);

  return {Eigen::Ref<Eigen::VectorXd>(workspace.qdldl_workspace.x, dim),
          kkt_error};
}

auto solve(const Input &input, const Settings &settings, Workspace &workspace,
           Output &output) -> void {
  for (int i = 0; i < s_dim; ++i) {
    assert(s[i] > 0.0 && z[i] > 0.0);
  }

  {
    ModelCallbackInput mci{
        .x = workspace.vars.x,
        .s = workspace.vars.s,
        .y = workspace.vars.y,
        .z = workspace.vars.z,
    };

    input.model_callback(mci, workspace.model_callback_output);
  }

  for (int iteration = 0; iteration < settings.max_iterations; ++iteration) {
    const int x_dim = model_callback_output.hessian_f.cols;
    const int s_dim = model_callback_output.jacobian_g.rows;
    const int y_dim = model_callback_output.jacobian_c.rows;

    const double mu =
        adaptive_mu(Eigen::Ref<Eigen::VectorXd>(workspace.vars.s, s_dim),
                    Eigen::Ref<Eigen::VectorXd>(workspace.vars.z, s_dim));

    const auto [dxsyz, kkt_error] = compute_search_direction(
        workspace.model_callback_output, settings, workspace.vars.s,
        workspace.vars.y, workspace.vars.z, mu, workspace);

    if (kkt_error < settings.max_kkt_violation) {
      output.exit_status = Status::SOLVED;
      return;
    }

    const double *dx = dxsyz;
    const double *ds = dxsyz + x_dim;
    const double *dz = dxsyz + x_dim + s_dim + y_dim;

    const double tau = std::max(settings.tau_min, mu > 0.0 ? 1.0 - mu : 0.0);

    const auto [alpha_s_max, alpha_z_max] = get_max_step_sizes(
        s_dim, tau, workspace.vars.s, workspace.vars.z, ds, dz);

    const double rho = get_rho(
        workspace.model_callback_output,
        Eigen::Ref<Eigen::VectorXd>(workspace.vars.x, x_dim),
        Eigen::Ref<Eigen::VectorXd>(workspace.vars.s, s_dim),
        Eigen::Ref<Eigen::VectorXd>(dx), Eigen::Ref<Eigen::VectorXd>(ds), mu);

    const double merit = merit_function(
        workspace.model_callback_output,
        Eigen::Ref<Eigen::VectorXd>(workspace.vars.x, x_dim),
        Eigen::Ref<Eigen::VectorXd>(workspace.vars.s, s_dim), mu, rho);

    const double merit_slope = merit_function_slope(
        workspace.model_callback_output,
        Eigen::Ref<Eigen::VectorXd>(workspace.vars.x, x_dim),
        Eigen::Ref<Eigen::VectorXd>(workspace.vars.s, s_dim),
        Eigen::Ref<Eigen::VectorXd>(dx), Eigen::Ref<Eigen::VectorXd>(ds), mu,
        rho);

    bool ls_succeeded = false;
    for (double alpha = alpha_s_max; alpha > settings.line_search_min_step_size;
         alpha *= settings.line_search_factor) {

      for (int i = 0; i < x_dim; ++i) {
        workspace.vars.next_x[i] = workspace.vars.x[i] + alpha * dx[i];
      }

      for (int i = 0; i < s_dim; ++i) {
        workspace.vars.next_s[i] = workspace.vars.s[i] + alpha * ds[i];
      }

      ModelCallbackInput mci{
          .x = workspace.vars.next_x,
          .s = workspace.vars.next_s,
          .y = workspace.vars.y,
          .z = workspace.vars.z,
      };
      input.model_callback(mci, workspace.model_callback_output);

      // TODO(joao): cache this into the next cycle!
      const double new_merit = merit_function(
          workspace.model_callback_output,
          Eigen::Ref<Eigen::VectorXd>(workspace.vars.next_x, x_dim),
          Eigen::Ref<Eigen::VectorXd>(workspace.vars.next_s, s_dim), mu, rho);

      if (new_merit - merit < settings.armijo_factor * merit_slope * alpha) {
        ls_succeeded = true;
        break;
      }
    }

    if (!ls_succeeded) {
      output.exit_status = Status::LINE_SEARCH_FAILURE;
      return;
    }
  }
}

} // namespace sip
