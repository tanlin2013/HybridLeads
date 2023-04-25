#include <list>

#include "hybridbasis/fixed_point_tensor.h"
#include "hybridbasis/gluon.h"
#include "hybridbasis/mpo_model.h"
#include "hybridbasis/utils.h"

int main(int argc, char* argv[]) {
  int n_left = 4;
  std::list<int> list_n_sys = {20, 40, 60, 80, 100, 120, 140, 160, 180, 200};
  int n_right = n_left;

  for (int n_sys : list_n_sys) {
    int n_tot = n_left + n_sys + n_right;
    Args model_args = {"t_left",      1.0, "t_left_sys", 1.0, "t_sys",       1.0,
                       "t_right_sys", 1.0, "t_right",    1.0, "mu_left",     0.0,
                       "mu_sys",      0.0, "mu_right",   0.0, "ConserveQNs", false};
    TightBinding model(n_left, n_sys, n_right, model_args);
    Args itdvp_args = {"dt", INFINITY, "max_bond_dim", 64};
    auto sites = Fermion(n_sys + 2, {"ConserveQNs", false});
    Gluon gluon(model.mpo(), sites, n_left - 1, n_right - 1, itdvp_args);

    auto sweeps = Sweeps(60);
    sweeps.maxdim() = 10, 20, 100, 200, 200;
    sweeps.cutoff() = 1E-8;
    auto [energy, psi] = dmrg(
        gluon.sys_mpo(), gluon.left_env(), gluon.right_env(),
        gluon.uniform_state(Left, length(model.sites()) / 2), sweeps, {"Silent", false}
    );

    // file pointer
    fstream fout;
    // opens an existing csv file or creates a new file.
    fout.open("data.csv", ios::out | ios::app);

    for (int site : range1(n_sys + 2 - 1)) {
      int bond_dim = dim(rightLinkIndex(psi, site));
      fout << n_sys + 2 << ", " << site << ", " << bond_dim << "\n";
    }
  }

  return 0;
}
