#include "sip/types.hpp"

#include <cassert>
#include <fmt/color.h>
#include <fmt/core.h>
#include <ostream>

namespace sip {

auto operator<<(std::ostream &os, Status const &status) -> std::ostream & {
  switch (status) {
  case Status::SOLVED:
    os << "SOLVED";
    break;
  case Status::SUBOPTIMAL:
    os << "SUBOPTIMAL";
    break;
  case Status::LOCALLY_INFEASIBLE:
    os << "LOCALLY_INFEASIBLE";
    break;
  case Status::ITERATION_LIMIT:
    os << "ITERATION_LIMIT";
    break;
  case Status::LINE_SEARCH_ITERATION_LIMIT:
    os << "LINE_SEARCH_ITERATION_LIMIT";
    break;
  case Status::LINE_SEARCH_FAILURE:
    os << "LINE_SEARCH_FAILURE";
    break;
  case Status::TIMEOUT:
    os << "TIMEOUT";
    break;
  case Status::FAILED_CHECK:
    os << "FAILED_CHECK";
    break;
  case Status::FACTORIZATION_FAILURE:
    os << "FACTORIZATION_FAILURE";
    break;
  }
  return os;
}

void VariablesWorkspace::reserve(int x_dim, int s_dim, int y_dim) {
  x = new double[x_dim];
  s = new double[s_dim];
  y = new double[y_dim];
  z = new double[s_dim];
}

void VariablesWorkspace::free() {
  delete[] x;
  delete[] s;
  delete[] y;
  delete[] z;
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

  return cum_size;
}

void MiscellaneousWorkspace::reserve(int s_dim) {
  g_plus_s = new double[s_dim];
}

void MiscellaneousWorkspace::free() { delete[] g_plus_s; }

auto MiscellaneousWorkspace::mem_assign(int s_dim, unsigned char *mem_ptr)
    -> int {
  int cum_size = 0;

  g_plus_s = reinterpret_cast<decltype(g_plus_s)>(mem_ptr + cum_size);
  cum_size += s_dim * sizeof(double);

  return cum_size;
}

void ComputeSearchDirectionWorkspace::reserve(int s_dim, int y_dim, int kkt_dim,
                                              int full_dim) {
  w = new double[s_dim];
  r2 = new double[y_dim];
  r3 = new double[s_dim];
  rhs_block_3x3 = new double[kkt_dim];
  sol_block_3x3 = new double[kkt_dim];
  iterative_refinement_error_sol = new double[kkt_dim];
  residual = new double[full_dim];
}

void ComputeSearchDirectionWorkspace::free() {
  delete[] w;
  delete[] r2;
  delete[] r3;
  delete[] rhs_block_3x3;
  delete[] sol_block_3x3;
  delete[] iterative_refinement_error_sol;
  delete[] residual;
}

auto ComputeSearchDirectionWorkspace::mem_assign(int s_dim, int y_dim,
                                                 int kkt_dim, int full_dim,
                                                 unsigned char *mem_ptr)
    -> int {
  int cum_size = 0;

  w = reinterpret_cast<decltype(w)>(mem_ptr + cum_size);
  cum_size += s_dim * sizeof(double);
  r2 = reinterpret_cast<decltype(r2)>(mem_ptr + cum_size);
  cum_size += y_dim * sizeof(double);
  r3 = reinterpret_cast<decltype(r3)>(mem_ptr + cum_size);
  cum_size += s_dim * sizeof(double);
  rhs_block_3x3 = reinterpret_cast<decltype(rhs_block_3x3)>(mem_ptr + cum_size);
  cum_size += kkt_dim * sizeof(double);
  sol_block_3x3 = reinterpret_cast<decltype(sol_block_3x3)>(mem_ptr + cum_size);
  cum_size += kkt_dim * sizeof(double);
  iterative_refinement_error_sol =
      reinterpret_cast<decltype(iterative_refinement_error_sol)>(mem_ptr +
                                                                 cum_size);
  cum_size += kkt_dim * sizeof(double);
  residual = reinterpret_cast<decltype(residual)>(mem_ptr + cum_size);
  cum_size += full_dim * sizeof(double);

  return cum_size;
}

void PenaltyParameterWorkspace::reserve(int s_dim, int y_dim) {
  y = new double[y_dim];
  z = new double[s_dim];
}

void PenaltyParameterWorkspace::free() {
  delete[] y;
  delete[] z;
}

auto PenaltyParameterWorkspace::mem_assign(int s_dim, int y_dim,
                                           unsigned char *mem_ptr) -> int {
  int cum_size = 0;

  y = reinterpret_cast<decltype(y)>(mem_ptr + cum_size);
  cum_size += y_dim * sizeof(double);

  z = reinterpret_cast<decltype(z)>(mem_ptr + cum_size);
  cum_size += s_dim * sizeof(double);

  return cum_size;
}

void FilterWorkspace::reserve(int filter_capacity) {
  theta = filter_capacity > 0 ? new double[filter_capacity] : nullptr;
  f = filter_capacity > 0 ? new double[filter_capacity] : nullptr;
  size = 0;
  capacity = filter_capacity;
}

void FilterWorkspace::free() {
  delete[] theta;
  delete[] f;
}

auto FilterWorkspace::mem_assign(int filter_capacity, unsigned char *mem_ptr)
    -> int {
  int cum_size = 0;
  theta = reinterpret_cast<decltype(theta)>(mem_ptr + cum_size);
  cum_size += filter_capacity * sizeof(double);
  f = reinterpret_cast<decltype(f)>(mem_ptr + cum_size);
  cum_size += filter_capacity * sizeof(double);
  size = 0;
  capacity = filter_capacity;
  return cum_size;
}

void Workspace::reserve(int x_dim, int s_dim, int y_dim,
                        const Settings &settings) {
  vars.reserve(x_dim, s_dim, y_dim);
  delta_vars.reserve(x_dim, s_dim, y_dim);
  next_vars.reserve(x_dim, s_dim, y_dim);
  nrhs.reserve(x_dim, s_dim, y_dim);
  miscellaneous_workspace.reserve(s_dim);
  const int kkt_dim = x_dim + s_dim + y_dim;
  const int full_dim = kkt_dim + s_dim;
  csd_workspace.reserve(s_dim, y_dim, kkt_dim, full_dim);
  penalties.reserve(s_dim, y_dim);
  filter.reserve(filter_capacity(settings));
}

void Workspace::free() {
  vars.free();
  delta_vars.free();
  next_vars.free();
  nrhs.free();
  miscellaneous_workspace.free();
  csd_workspace.free();
  penalties.free();
  filter.free();
}

auto Workspace::mem_assign(int x_dim, int s_dim, int y_dim,
                           const Settings &settings, unsigned char *mem_ptr)
    -> int {
  int cum_size = 0;

  cum_size += vars.mem_assign(x_dim, s_dim, y_dim, mem_ptr + cum_size);
  cum_size += delta_vars.mem_assign(x_dim, s_dim, y_dim, mem_ptr + cum_size);
  cum_size += next_vars.mem_assign(x_dim, s_dim, y_dim, mem_ptr + cum_size);
  cum_size += nrhs.mem_assign(x_dim, s_dim, y_dim, mem_ptr + cum_size);
  cum_size += miscellaneous_workspace.mem_assign(s_dim, mem_ptr + cum_size);
  const int kkt_dim = x_dim + s_dim + y_dim;
  const int full_dim = kkt_dim + s_dim;
  cum_size += csd_workspace.mem_assign(s_dim, y_dim, kkt_dim, full_dim,
                                       mem_ptr + cum_size);
  cum_size += penalties.mem_assign(s_dim, y_dim, mem_ptr + cum_size);
  cum_size += filter.mem_assign(filter_capacity(settings), mem_ptr + cum_size);

  return cum_size;
}

} // namespace sip
