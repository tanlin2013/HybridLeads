#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/trompeloeil.hpp>

#include "hybridbasis/mpo_model.h"
#include "hybridbasis/utils.h"
#include "itensor/all.h"
#include "kbasis/OneParticleBasis.h"

using namespace itensor;  // NOLINT
using namespace Catch;    // NOLINT

TEST_CASE("Check single particle hamiltonian", "[TestTightBindingSingleParticleHam]") {
  int n_left = GENERATE(4, 6);
  int n_sys = GENERATE(2, 4);
  int n_right = n_left;
  int n_tot = n_left + n_sys + n_right;
  Real t = 0.5;
  Real mu = GENERATE(0.0, 0.1);

  auto elems = tight_binding_Hamilt(n_tot, t, mu);
  arma::mat expected_ham(&elems(0, 0), n_tot, n_tot);

  Args args = {"t_left",  t, "t_left_sys", t,  "t_sys",  t,  "t_right_sys", t,
               "t_right", t, "mu_left",    mu, "mu_sys", mu, "mu_right",    mu};
  TightBinding model(n_left, n_sys, n_right, args);
  arma::mat ham = model.single_particle_ham();
  arma::umat elem_wise_comparison = (ham == expected_ham);
  CHECK(arma::all(arma::vectorise(elem_wise_comparison) == true));
}

TEST_CASE("Check single particle basis rotation", "[TestBasisRotation]") {
  int n_left = GENERATE(4, 6);
  int n_sys = GENERATE(2, 4);
  int n_right = n_left;
  int n_tot = n_left + n_sys + n_right;
  Real t1 = 0.5;
  Real t2 = GENERATE(0.1, 0.5, 1.0);
  Real mu_left = GENERATE(0.0, 0.1);
  Real mu_sys = GENERATE(0.0, 0.1);
  Real mu_right = GENERATE(0.0, 0.1);
  Args args = {"t_left",      t1,     "t_left_sys", t2,      "t_sys",   t1,
               "t_right_sys", t2,     "t_right",    t1,      "mu_left", mu_left,
               "mu_sys",      mu_sys, "mu_right",   mu_right};
  TightBinding model(n_left, n_sys, n_right, args);
  arma::mat ham = model.single_particle_ham();
  arma::mat hybrid_ham = model.hybrid_basis_ham();

  CHECK_THAT(arma::trace(ham), Matchers::WithinAbs(arma::trace(hybrid_ham), 1e-12));
  CHECK(std::abs(arma::det(ham)) > 1e-12);  // check non-singular
  CHECK_THAT(arma::det(ham), Matchers::WithinAbs(arma::det(hybrid_ham), 1e-12));
}

TEST_CASE("Check MPO on real space parts", "[TestTightBindingMPO]") {
  int n_left = GENERATE(4, 6);
  int n_sys = GENERATE(2, 4);
  int n_right = n_left;
  int n_tot = n_left + n_sys + n_right;
  Args args = {"t_left",  0.5, "t_left_sys", 0.5, "t_sys",  0.5, "t_right_sys", 0.5,
               "t_right", 0.5, "mu_left",    1.0, "mu_sys", 0.5, "mu_right",    0.1};
  TightBinding model(n_left, n_sys, n_right, args);
  auto H = model.mpo();
  for (int i = 2; i < n_left - 1; ++i) {
    CHECK(ALLCLOSE(H(i), H(i + 1)) == true);
  }
  for (int i = n_left + n_sys + 2; i < n_tot - 1; ++i) {
    CHECK(ALLCLOSE(H(i), H(i + 1)) == true);
  }
}

TEST_CASE("Check ground state energies are consistent", "[TestTightBindingGS]") {
  int n_left = GENERATE(4, 6);
  int n_sys = GENERATE(2, 4);
  int n_right = n_left;
  int n_tot = n_left + n_sys + n_right;
  Real t1 = 0.5;
  Real t2 = GENERATE(0.1, 0.5, 1.0);
  Real mu_left = GENERATE(0.0, 0.1);
  Real mu_sys = GENERATE(0.0, 0.1);
  Real mu_right = GENERATE(0.0, 0.1);
  Args args = {"t_left",      t1,     "t_left_sys", t2,      "t_sys",   t1,
               "t_right_sys", t2,     "t_right",    t1,      "mu_left", mu_left,
               "mu_sys",      mu_sys, "mu_right",   mu_right};
  TightBinding model(n_left, n_sys, n_right, args);

  // Create a random starting MPS
  auto state = InitState(model.sites());
  for (auto i : range1(n_tot)) {
    if (i % 2 == 1)
      state.set(i, "1");
    else
      state.set(i, "0");
  }
  auto psi0 = randomMPS(state);

  // Run DMRG in 2 bases respectively
  auto sweeps = Sweeps(30);
  sweeps.maxdim() = 8, 16, 32, 64, 64;
  sweeps.cutoff() = 1E-10;
  auto [energy, psi] = dmrg(model.mpo(), psi0, sweeps, {"Silent", true});
  auto [expected_energy, expected_psi] =
      dmrg(model.real_space_mpo(), psi0, sweeps, {"Silent", true});
  CHECK_THAT(energy, Matchers::WithinAbs(expected_energy, 1e-6));
}
