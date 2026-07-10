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
      FieldAttributes("x1"),     //
      FieldAttributes("x2"),     //
      FieldAttributes("rxn"),    //
      FieldAttributes("lap_n"), //
      FieldAttributes("x_total"),//
      FieldAttributes("Gamma"),
      FieldAttributes("x_Ni") //
  };

  SolveBlock main_fields(0, Explicit, Initialized, {0, 1, 2});
  main_fields.dependencies_rhs =
      make_dependency_set(fields, {"old_1(n)", "old_1(x1)", "old_1(x2)", "grad(old_1(n))",
                                   "grad(old_1(x1))", "grad(old_1(x2))", "old_1(rxn)", "old_1(lap_n)"});

  SolveBlock lap_n(1, Explicit, Uninitialized, {4});
  lap_n.dependencies_rhs = make_dependency_set(fields, {"grad(n)"});

  SolveBlock rxn(2, Explicit, Uninitialized, {3});
  rxn.dependencies_rhs = make_dependency_set(fields, {"n", "grad(n)", "x1", "x2", "lap_n"});

  SolveBlock pp(3, Explicit, PostProcess, {5, 6, 7});
  pp.dependencies_rhs = make_dependency_set(fields, {"n", "grad(n)", "x1", "x2", "lap_n"});

  std::vector<SolveBlock> solve_blocks({main_fields, lap_n, rxn, pp});

  UserInputParameters<dim> user_inputs(cli_options.get_parameters_filename());
  for (const auto &name : {"x1", "x2"})
    {
      double domain_size = user_inputs.spatial_discretization.rectangular_mesh.size.norm();
      user_inputs.spatial_discretization.refinement_criteria[name].criterion =
          RefinementFlags::Gradient;
      user_inputs.spatial_discretization.refinement_criteria[name].gradient_lower_bound =
          0.1 / domain_size;
    }
  PhaseFieldTools<dim>           pf_tools;
  CustomPDE<dim, degree, double> pde_operator(user_inputs, pf_tools);
  Problem<dim, degree, double>   problem(fields, solve_blocks, user_inputs, pf_tools, pde_operator);
  problem.solve();

  return 0;
}
