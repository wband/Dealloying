// SPDX-FileCopyrightText: © 2025 Xander Mensah
// SPDX-License-Identifier: GNU Lesser General Public Version 2.1

#include <prismspf/core/pde_operator_base.h>

#include <complex.h>
#include <conversion.h>

#include <dual.h>
#include <variation.h>

#include <ni-cr.h>

PRISMS_PF_BEGIN_NAMESPACE

template <unsigned int dim, unsigned int degree, typename number>
class CustomPDE : public PDEOperatorBase<dim, degree, number>
{
public:
  using ScalarValue     = dealii::VectorizedArray<number>;
  using ScalarGrad      = dealii::Tensor<1, dim, ScalarValue>;
  using ScalarHess      = dealii::Tensor<2, dim, ScalarValue>;
  using ScalarField     = Dual<ScalarValue, dim>;
  using ScalarVariation = Variation<ScalarValue, dim>;

  using VectorValue     = dealii::Tensor<1, dim, ScalarValue>;
  using VectorGrad      = dealii::Tensor<2, dim, ScalarValue>;
  using VectorHess      = dealii::Tensor<3, dim, ScalarValue>;
  using VectorField     = Dual<VectorValue, dim>;
  using VectorVariation = Variation<VectorValue, dim>;

  using ComplexValue     = Complex<ScalarValue>;
  using ComplexGrad      = dealii::Tensor<1, dim, ComplexValue>;
  using ComplexHess      = dealii::Tensor<2, dim, ComplexValue>;
  using ComplexField     = Dual<ComplexValue, dim>;
  using ComplexVariation = Variation<ComplexValue, dim>;

  using PDEOperatorBase<dim, degree, number>::get_user_inputs;
  using PDEOperatorBase<dim, degree, number>::get_pf_tools;

  // Hard coded
  number RT     = 1023.15 * 8.314; // J
  number Vmfact = 7.0e3;           // number of m*m*nm sheets per mol
  number gamma  = 2.0;             // J/m^2

  // Parameter file
  number deltaG0;
  number D1;
  number D1s;
  number D2;
  number j0;
  number l_int;
  number x1_init;
  number x2_init;
  number epsilon; // small number to avoid division by zero
  int int_delta;
  bool x_in_rxn;
  // NiCr Thermo
  NiCrThermo::Isothermal nicr_energy;

  /**
   * @brief Constructor.
   */
  explicit CustomPDE(const UserInputParameters<dim> &_user_inputs, PhaseFieldTools<dim> &_pf_tools)
      : PDEOperatorBase<dim, degree, number>(_user_inputs, _pf_tools),
        deltaG0(get_user_inputs().user_constants.get_double("deltaG0")),
        D1(get_user_inputs().user_constants.get_double("D1")),
        D1s(get_user_inputs().user_constants.get_double("D1s")),
        D2(get_user_inputs().user_constants.get_double("D2")),
        j0(get_user_inputs().user_constants.get_double("j0")),
        l_int(get_user_inputs().user_constants.get_double("l_int")),
        x1_init(get_user_inputs().user_constants.get_double("x1_init")),
        x2_init(get_user_inputs().user_constants.get_double("x2_init")),
        epsilon(get_user_inputs().user_constants.get_double("epsilon")),
        int_delta(get_user_inputs().user_constants.get_int("int_delta")),
        x_in_rxn(get_user_inputs().user_constants.get_bool("x_in_rxn"))
  {
    nicr_energy.set_temperature(RT / NiCrThermo::R);
  }

private:
  void
  set_initial_condition([[maybe_unused]] const unsigned int       &index,
                        [[maybe_unused]] const unsigned int       &component,
                        [[maybe_unused]] const dealii::Point<dim> &point,
                        [[maybe_unused]] number                   &scalar_value,
                        [[maybe_unused]] number &vector_component_value) const override
  {
    // Custom coordinate system
    const dealii::Tensor<1, dim> &mesh_size =
        get_user_inputs().spatial_discretization.rectangular_mesh.size;
    const dealii::Point<dim>      center(mesh_size / 2.0);
    const dealii::Point<dim>      p(point);
    [[maybe_unused]] const double x  = (dim > 0) ? p[0] : 0.;
    [[maybe_unused]] const double y  = (dim > 1) ? p[1] : 0.;
    [[maybe_unused]] const double z  = (dim > 2) ? p[2] : 0.;
    [[maybe_unused]] const double lx = (dim > 0) ? mesh_size[0] : 0.;
    [[maybe_unused]] const double ly = (dim > 1) ? mesh_size[1] : 0.;
    [[maybe_unused]] const double lz = (dim > 2) ? mesh_size[2] : 0.;
    // ===========================================================================
    // FUNCTION FOR INITIAL CONDITIONS
    // ===========================================================================
    using std::cos;
    using std::max;
    using std::min;
    using std::sin;
    using std::sqrt;
    using std::tanh;
    if (index == 0)
      {
        double dist  = x - lx / 2.0;
        double phi   = interface(dist);
        scalar_value = max(min(phi, 1.0 - epsilon), epsilon);
        return;
      }
    if (index == 1)
      {
        scalar_value = x1_init;
        return;
      }
    if (index == 2)
      {
        scalar_value = x2_init;
        return;
      }
    if (index > 2)
      {
        scalar_value = 0.0;
        return;
      }
  }
  void
  set_dirichlet([[maybe_unused]] const unsigned int       &index,
                [[maybe_unused]] const unsigned int       &boundary_id,
                [[maybe_unused]] const unsigned int       &component,
                [[maybe_unused]] const dealii::Point<dim> &point,
                [[maybe_unused]] number                   &scalar_value,
                [[maybe_unused]] number                   &vector_component_value) const override
  {
    scalar_value = x2_init;
  }

  void
  compute_rhs([[maybe_unused]] FieldContainer<dim, degree, number> &variable_list,
              [[maybe_unused]] const SimulationTimer               &sim_timer,
              [[maybe_unused]] unsigned int                         solve_block_id) const override
  {
    const dealii::Tensor<1, dim> &mesh_size =
        get_user_inputs().spatial_discretization.rectangular_mesh.size;
    const ScalarValue dx = mesh_size[0]/
         (get_user_inputs().spatial_discretization.rectangular_mesh.subdivisions[0]
         *std::pow(2.0, get_user_inputs().spatial_discretization.global_refinement));
    constexpr double pi = 3.14159265359;
    const number     dt = sim_timer.get_timestep();
    if (solve_block_id == 0) // n, x
      {
        const ScalarValue n       = variable_list.template get_value<Scalar, OldOne>(0);
        const ScalarValue x1      = variable_list.template get_value<Scalar, OldOne>(8);
        const ScalarValue x2      = variable_list.template get_value<Scalar, OldOne>(9);
        const ScalarValue rxn     = variable_list.template get_value<Scalar, OldOne>(3);
        // n
        variable_list.set_value_term(0, n + dt * rxn);

        // x1
        variable_list.set_value_term(1, x1 + dt * rxn * (1.0 - x1) / n);

        // x2
        variable_list.set_value_term(2, x2 - dt * rxn * (1.0 - x2) / (1.0 - n));
      }
    else if (solve_block_id == 1) // x diffusion
      {
        const ScalarValue n       = variable_list.template get_value<Scalar, Current>(0);
        const ScalarGrad  n_grad  = variable_list.template get_gradient<Scalar, Current>(0);
        const ScalarValue x1      = variable_list.template get_value<Scalar, Current>(1);
        const ScalarGrad  x1_grad = variable_list.template get_gradient<Scalar, Current>(1);
        const ScalarValue x2      = variable_list.template get_value<Scalar, Current>(2);
        const ScalarGrad  x2_grad = variable_list.template get_gradient<Scalar, Current>(2);
        // x1
        variable_list.set_value_term(
            8, x1 + dt * (D1 * x1_grad * n_grad
                      + 8.0 / l_int * (1.0 - n) * D1s * x1_grad * n_grad)/n);
        variable_list.set_gradient_term(
            8, dt * (-(D1 + 8.0 / l_int * (1.0 - n) * D1s) * x1_grad));

        // x2
        variable_list.set_value_term(9, x2 + dt * (-D2 * x2_grad * n_grad) / (1.0 - n));
        variable_list.set_gradient_term(9, dt * (-D2 * x2_grad));
      }
    else if (solve_block_id == 2) // deltaG
      {
        const ScalarValue n      = variable_list.template get_value<Scalar, Current>(0);
        const ScalarGrad  n_grad = variable_list.template get_gradient<Scalar, Current>(0);
        const ScalarValue x1     = variable_list.template get_value<Scalar, Current>(8);
        const ScalarValue x2     = variable_list.template get_value<Scalar, Current>(9);

        // deltaG
        const Dual<ScalarValue, 1> x1_dual(x1, dealii::Tensor<1, 1, ScalarValue>({1.0}));
        const Dual<ScalarValue, 1> x2_dual(x2, dealii::Tensor<1, 1, ScalarValue>({1.0}));
        const Dual<ScalarValue, 1> Gm_a = Gm_alpha(x1_dual);
        const Dual<ScalarValue, 1> Gm_b = Gm_beta(x2_dual);

        ScalarValue       mu1_chem = Gm_a.val + (1.0 - x1) * Gm_a.grad[0];
        ScalarValue       mu2_chem = Gm_b.val + (1.0 - x2) * Gm_b.grad[0];
        const ScalarValue deltaG_val =
            (mu2_chem - mu1_chem) / RT + 4.0 * Vmfact * gamma / RT / l_int * (2.0 * n - 1.0);

        // const ScalarValue deltaG_val = std::log(x2 / x1 ) + deltaG0 +
        //                                4.0 * Vmfact * gamma / RT / l_int * (2.0 * n - 1.0);
        variable_list.set_value_term(4, deltaG_val);
        variable_list.set_gradient_term(4, -n_grad * 8.0 * Vmfact * gamma / RT * l_int / (pi * pi));
      }
    else if (solve_block_id == 3) // rxn
      {
        number      upper(1.0 - epsilon);
        number      lower(epsilon);
        ScalarValue n      = variable_list.template get_value<Scalar, Current>(0);
        ScalarGrad  n_grad = variable_list.template get_gradient<Scalar, Current>(0);
        const ScalarValue deltaG = variable_list.template get_value<Scalar, Current>(4);
        const ScalarValue x1     = variable_list.template get_value<Scalar, Current>(8);
        const ScalarValue x2     = variable_list.template get_value<Scalar, Current>(9);
        ScalarValue x_prefact(1.0);
        if (x_in_rxn)
          {
            x_prefact = x1 * x2/0.01;
          }
        ScalarValue rxn_val;
        switch (int_delta)
          {
          case 0:
            rxn_val = -n_grad.norm_square() * n * (1.0 - n) *
               (128.0 * l_int / (pi * pi) / 3.0) * Vmfact * j0 * x_prefact * (-deltaG);
            break;
          case 1:
            rxn_val = -n_grad.norm() * Vmfact * j0 * x_prefact * (-deltaG);
            break;
          case 2:
            rxn_val = -n_grad.norm_square() * (8.0 * l_int/(pi*pi)) 
               * Vmfact * j0 * x_prefact * (-deltaG);
            break;
          default:
            rxn_val = -n_grad.norm_square() * n * (1.0 - n) *
               (128.0 * l_int / (pi * pi) / 3.0) * Vmfact * j0 * x_prefact * (-deltaG);
            break;
          }
        constrain_dvaldt(n, rxn_val, dt, lower, upper);
        variable_list.set_value_term(3, rxn_val);
      }
    else if (solve_block_id == 4) // pp
      {
        const ScalarValue n  = variable_list.template get_value<Scalar, Current>(0);
        const ScalarGrad  n_grad = variable_list.template get_gradient<Scalar, Current>(0);
        const ScalarValue x1 = variable_list.template get_value<Scalar, Current>(8);
        const ScalarValue x2 = variable_list.template get_value<Scalar, Current>(9);
        variable_list.set_value_term(5, n * x1 + (1.0 - n) * x2);
        variable_list.set_value_term(6, 4.0 * gamma/l_int * (n * (1.0 - n)
                             + l_int * l_int / (pi * pi) * n_grad.norm_square()));
        variable_list.set_value_term(7, n * (1.0 - x1));
      }
  }

  template <typename real>
  real
  Gm_alpha(const real &xB) const
  {
    // return nicr_energy.G_fcc(xB);
    using std::log;
    real xA = 1.0 - xB;
    real G(0.0);
    G += RT * (xB * log(xB) + xA * log(xA));
    return G;
  }

  template <typename real>
  real
  Gm_beta(const real &xB) const
  {
    using std::log;
    real xA = 1.0 - xB;
    real G(-deltaG0 * RT);
    G += RT * (xB * log(xB) + xA * log(xA));
    return G;
  }

  void
  pre_solve_block([[maybe_unused]] SolveContext<dim, degree, number> &solve_context,
                  [[maybe_unused]] unsigned int                       solver_id) override
  {
    // solve_context.get_simulation_timer().get_increment();
    // solve_context.get_user_inputs().spatial_discretization.has_adaptivity;
  }

  void
  post_solve_block([[maybe_unused]] SolveContext<dim, degree, number> &solve_context,
                   [[maybe_unused]] unsigned int                       solver_id) override
  {}

private:
  template <typename num>
  void
  constrain_dvaldt(const num &val, num &dvaldt, double dt, double lower = 0.0,
                   double upper = 1.0) const
  {
    using std::max;
    using std::min;
    num top = max(val + dvaldt * dt, num(upper));
    num bot = min(val + dvaldt * dt, num(lower));
    dvaldt  = (dvaldt * dt + (upper - top - (bot - lower))) / dt;
  }

  /**
   *@brief return the double obstacle interface function
   */
  template <typename real>
  const real
  interface(const real &x) const
  {
    using std::max;
    using std::min;
    using std::sin;
    constexpr double pi = 3.14159265359;
    return 0.5 * (1.0 + sin(pi * max(-0.5, min(0.5, x / l_int))));
  }
};

PRISMS_PF_END_NAMESPACE
