// SPDX-FileCopyrightText: © 2025 PRISMS Center at the University of Michigan
// SPDX-License-Identifier: GNU Lesser General Public Version 2.1

#include "custom_pde.h"

#include <prismspf/core/field_attributes.h>
#include <prismspf/core/parse_cmd_options.h>
#include <prismspf/core/problem.h>
#include <prismspf/core/solve_block.h>

using namespace prismspf;

int
main(int argc, char *argv[])
{
  // Initialize MPI
  dealii::Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv,
                                                    dealii::numbers::invalid_unsigned_int);

  // Restrict deal.II console printing
  dealii::deallog.depth_console(0);

  // Parse the command line options (if there are any) to get the name of the
  // input file
  ParseCMDOptions cli_options(argc, argv);

  constexpr unsigned int dim    = 2;
  constexpr unsigned int degree = 1;

  std::vector<FieldAttributes> fields = {
      FieldAttributes("n"),      //
      FieldAttributes("x1_rxn"),     //
      FieldAttributes("x2_rxn"),     //
      FieldAttributes("rxn"),    //
      FieldAttributes("deltaG"), //
      FieldAttributes("x_total"),//
      FieldAttributes("Gamma"),
      FieldAttributes("x_Ni"), //
      FieldAttributes("x1_diff"),     //
      FieldAttributes("x2_diff")     //
  };

  SolveBlock main_fields(0, Explicit, Initialized, {0, 1, 2});
  main_fields.dependencies_rhs =
      make_dependency_set(fields, {"old_1(n)", "old_1(x1_diff)", 
         "old_1(x2_diff)", "old_1(rxn)"});

  SolveBlock diffusion(1, Explicit, Uninitialized, {8, 9});
  diffusion.dependencies_rhs =
      make_dependency_set(fields, {"n", "x1_rxn", "x2_rxn", "grad(n)",
         "grad(x1_rxn)", "grad(x2_rxn)"});

  SolveBlock deltaG(2, Explicit, Uninitialized, {4});
  deltaG.dependencies_rhs = make_dependency_set(fields, {"x1_diff", "x2_diff", "n", "grad(n)"});

  SolveBlock rxn(3, Explicit, Uninitialized, {3});
  rxn.dependencies_rhs = make_dependency_set(fields, {"x1_diff", "x2_diff", "n", "grad(n)", "deltaG"});

  SolveBlock pp(4, Explicit, PostProcess, {5, 6, 7});
  pp.dependencies_rhs = make_dependency_set(fields, {"n", "grad(n)", "x1_diff", "x2_diff"});

  std::vector<SolveBlock> solve_blocks({main_fields, diffusion, deltaG, rxn, pp});

  UserInputParameters<dim> user_inputs(cli_options.get_parameters_filename());
  PhaseFieldTools<dim>           pf_tools;
  CustomPDE<dim, degree, double> pde_operator(user_inputs, pf_tools);
  Problem<dim, degree, double>   problem(fields, solve_blocks, user_inputs, pf_tools, pde_operator);
  problem.solve();

  return 0;
}
