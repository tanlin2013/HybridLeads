#include <catch2/catch_all.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/trompeloeil.hpp>
#include <list>

#include "hybridbasis/fixed_point_tensor.h"
#include "hybridbasis/gluon.h"
#include "hybridbasis/mpo_model.h"
#include "hybridbasis/utils.h"

using namespace itensor;  // NOLINT
using namespace Catch;    // NOLINT

TEST_CASE("Check gluon indices", "[TestCommonInds]") {
  int n_left = 4;
  int n_sys = GENERATE(2, 4, 6);
  int n_right = n_left;
  int n_tot = n_left + n_sys + n_right;
  Args model_args = {"t_left",      0.5,  "t_left_sys", 0.01, "t_sys",       0.5,
                     "t_right_sys", 0.01, "t_right",    0.5,  "mu_left",     0.0,
                     "mu_sys",      0.0,  "mu_right",   0.0,  "ConserveQNs", false};
  TightBinding model(n_left, n_sys, n_right, model_args);
  auto mpo = model.mpo();
  Args itdvp_args = {"max_bond_dim", 6};
  auto sites = Fermion(n_sys + 2, {"ConserveQNs", false});

  Gluon gluon(mpo, sites, n_left - 1, n_right - 1, itdvp_args);
  auto sys_mpo = gluon.sys_mpo();
  auto left_env = gluon.left_env();
  auto right_env = gluon.right_env();
  auto rand_init_state = gluon.random_init_state();
  auto uni_init_state = gluon.uniform_state(Left, n_sys / 2);

  CHECK(order(sys_mpo(1)) == 4);
  CHECK(order(sys_mpo(length(sites))) == 4);
  CHECK(order(rand_init_state(1)) == 3);
  CHECK(order(rand_init_state(length(sites))) == 3);
  for (int site : itensor::range1(length(sites))) {
    CHECK(order(uni_init_state(site)) == 3);
  }

  std::list<IndexSet> virtual_indices = {
      commonInds(sys_mpo(1), left_env),
      commonInds(sys_mpo(length(sites)), right_env),
      commonInds(rand_init_state(1), left_env),
      commonInds(rand_init_state(length(sites)), right_env),
      commonInds(uni_init_state(1), left_env),
      commonInds(uni_init_state(length(sites)), right_env),
  };

  for (IndexSet idx : virtual_indices) {
    CHECK(length(idx) == 1);
    CHECK(hasTags(idx(1), "Link"));
  }

  for (int site : itensor::range1(length(sites))) {
    IndexSet phys_inds = commonInds(sys_mpo(site), uni_init_state(site));
    CHECK(length(phys_inds) == 1);
    CHECK(hasTags(phys_inds(1), "Site"));
  }
}

TEST_CASE("Compare ground state energy density", "[TestEnergyDensity]") {
  int n_left = 4;
  int n_sys = GENERATE(20, 40, 60, 80, 100, 120, 140, 160, 180, 200);
  int n_right = n_left;
  int n_tot = n_left + n_sys + n_right;
  Args model_args = {"t_left",      1.0, "t_left_sys", 1.0, "t_sys",       1.0,
                     "t_right_sys", 1.0, "t_right",    1.0, "mu_left",     0.0,
                     "mu_sys",      0.0, "mu_right",   0.0, "ConserveQNs", false};
  TightBinding model(n_left, n_sys, n_right, model_args);
  Args itdvp_args = {"dt", INFINITY, "max_bond_dim", 64};
  auto sites = Fermion(n_sys + 2, {"ConserveQNs", false});
  Gluon gluon(model.mpo(), sites, n_left - 1, n_right - 1, itdvp_args);

  // auto state = InitState(model.sites());
  // for (auto i : range1(n_tot)) {
  //   if (i % 2 == 1)
  //     state.set(i, "1");
  //   else
  //     state.set(i, "0");
  // }
  // auto psi0 = randomMPS(state);

  auto sweeps = Sweeps(60);
  sweeps.maxdim() = 10, 20, 100, 200, 200;
  sweeps.cutoff() = 1E-8;
  auto [energy, psi] = dmrg(
      gluon.sys_mpo(), gluon.left_env(), gluon.right_env(),
      gluon.uniform_state(Left, length(model.sites()) / 2), sweeps, {"Silent", true}
  );
  // auto [expected_energy, expected_psi] =
  //     dmrg(model.mpo(), psi0, sweeps, {"Silent", true});

  // std::cout << energy / (n_sys + 2) << ", " << n_sys + 2 << ", "
  //           << expected_energy / n_tot << ", " << n_tot << std::endl;
  //   CHECK_THAT(energy, Matchers::WithinAbs(expected_energy, 1e-6));

  // int center_site = length(model.sites()) / 2;
  // psi.position(center_site);
  // PrintData(psi);

  // auto l = leftLinkIndex(psi, center_site);
  // auto s = siteIndex(psi, center_site);
  // auto [U,S,V] = svd(psi(center_site),{l,s});
  // auto u = commonIndex(U,S);

  // Real entropy = 0.;
  // for(auto n : range1(dim(u))) {
  //   auto Sn = elt(S,n,n);
  //   auto p = sqr(Sn);
  //   if(p > 1E-12) entropy += -p*log(p);
  // }

  // file pointer
  fstream fout;
  // opens an existing csv file or creates a new file.
  fout.open("data.csv", ios::out | ios::app);

  for (int site : range1(n_sys + 2 - 1)) {
    int bond_dim = dim(rightLinkIndex(psi, site));
    fout << n_sys + 2 << ", " << site << ", " << bond_dim << "\n";
  }
}
