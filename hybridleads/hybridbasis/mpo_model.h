#ifndef HYBRIDLEADS_HYBRIDBASIS_MPO_MODEL_H_
#define HYBRIDLEADS_HYBRIDBASIS_MPO_MODEL_H_

#include <armadillo>
#include <tuple>

#include "itensor/all.h"
#include "kbasis/OneParticleBasis.h"

using Real = itensor::Real;

class MPOModel {
 public:
  /**
   * @brief Base class of MPO model in the hybrid basis jointed by left lead
   * (real), system (k), and right lead (real).
   *
   * @param left_size
   * @param system_size
   * @param right_size
   * @throws `std::invalid_argument` when lead size is less than 4.
   * @note User should implement the virtual member functions in child class.
   */
  MPOModel(int left_size, int system_size, int right_size) {
    if ((left_size < 4) || (right_size < 4)) {
      throw std::invalid_argument("Lead size should be equal or greater than 4.");
    }
    n_left_ = left_size;
    n_sys_ = system_size;
    n_right_ = right_size;
    n_tot_ = left_size + system_size + right_size;
  }

  /**
   * @brief Return the size of each block in joint system.
   *
   * @returns std::tuple<int, int, int> (left_size, system_size, right_size).
   */
  std::tuple<int, int, int> sizes() {
    return std::make_tuple(n_left_, n_sys_, n_right_);
  }

  /**
   * @brief Return `SiteSet` of this model.
   *
   * @returns itensor::SiteSet
   */
  itensor::SiteSet sites() { return sites_; }

  /**
   * @brief Return the `MPO` instance in hybrid basis.
   *
   * @returns itensor::MPO
   */
  itensor::MPO mpo() { return itensor::toMPO(ampo_); }

  /**
   * @brief Return the `MPO` instance in real-space basis.
   *
   * @return itensor::MPO
   */
  itensor::MPO real_space_mpo() { return itensor::toMPO(rs_ampo_); }

  /**
   * @brief Return the single particle Hamiltonian in "standard" basis.
   *
   * @returns arma::mat - sp_ham.
   */
  arma::mat single_particle_ham() { return sp_ham_; }

  /**
   * @brief Return the single particle Hamiltonian in hybrid basis.
   *
   * @returns arma::mat - hybrid_ham.
   */
  arma::mat hybrid_basis_ham() { return hybrid_ham_; }

 protected:
  int n_left_, n_sys_, n_right_, n_tot_;
  itensor::SiteSet sites_;
  itensor::AutoMPO ampo_, rs_ampo_;
  arma::mat sp_ham_;
  arma::mat hybrid_ham_;

  /**
   * @brief Rotate the basis of single particle Hamiltonian into the hybrid
   * basis.
   */
  void basis_transformer() {
    if (arma::rank(sp_ham_) < n_tot_) {
      throw std::runtime_error("The single particle Hamiltonian is probably singular.");
    }
    arma::vec evals;
    arma::mat evecs;
    arma::mat sub_ham_mat =
        sp_ham_.submat(n_left_, n_left_, n_left_ + n_sys_ - 1, n_left_ + n_sys_ - 1);
    arma::eig_sym(evals, evecs, sub_ham_mat);
    arma::mat unitary_mat = arma::eye(n_tot_, n_tot_);
    unitary_mat.submat(n_left_, n_left_, n_left_ + n_sys_ - 1, n_left_ + n_sys_ - 1) =
        evecs;
    hybrid_ham_ = unitary_mat.t() * sp_ham_ * unitary_mat;
  }

  /**
   * @brief Abstract method for generating single-particle Hamiltonian.
   *
   * @note Please implement this method in the child class.
   */
  virtual void gen_single_particle_ham() {}

  /**
   * @brief Abstract method for generating `AutoMPO` in hybrid basis.
   *
   * @note Please implement this method in the child class.
   */
  virtual void gen_auto_mpo() {}

  /**
   * @brief Abstract method for generating `AutoMPO` in real-space basis.
   *
   * @note Please implement this method in the child class.
   */
  virtual void gen_real_space_auto_mpo() {}
};

class TightBinding : public MPOModel {
 public:
  /**
   * @brief Construct the tight-binding model in hybridbasis.
   *
   * @param left_size
   * @param system_size
   * @param right_size
   * @param args Arguments containing the model parameters with these keywords:
   * @param t_left (Real), default 0.
   * @param t_left_sys (Real), default 0.
   * @param t_sys (Real), default 0.
   * @param t_right_sys (Real), default 0.
   * @param t_right (Real), default 0.
   * @param mu_left (Real), default 0.
   * @param mu_sys (Real), default 0.
   * @param mu_right (Real), default 0.
   * @param ConserveQNs (bool), default false. This arg is passed to class:
   * `ITensor::FermionSite`.
   * @example
   * \code
   * Args args = {"t_left",  0.5};
   * \endcode
   */
  TightBinding(
      int left_size, int system_size, int right_size,
      itensor::Args const& args = itensor::Args::global()
  )
      : MPOModel(left_size, system_size, right_size) {
    t_left_ = args.getReal("t_left", 0.0);
    t_left_sys_ = args.getReal("t_left_sys", 0.0);
    t_sys_ = args.getReal("t_sys", 0.0);
    t_right_sys_ = args.getReal("t_right_sys", 0.0);
    t_right_ = args.getReal("t_right", 0.0);
    mu_left_ = args.getReal("mu_left", 0.0);
    mu_sys_ = args.getReal("mu_sys", 0.0);
    mu_right_ = args.getReal("mu_right", 0.0);
    bool conserve_qns = args.getBool("ConserveQNs", false);
    sites_ = itensor::Fermion(n_tot_, {"ConserveQNs", conserve_qns});
    ampo_ = itensor::AutoMPO(sites_);
    rs_ampo_ = itensor::AutoMPO(sites_);
    gen_single_particle_ham();
    basis_transformer();
    gen_auto_mpo();
    gen_real_space_auto_mpo();
  }

 protected:
  Real t_left_, t_left_sys_, t_sys_, t_right_sys_, t_right_;
  Real mu_left_, mu_sys_, mu_right_;

  arma::mat block_tight_binding_ham(int n, Real t, Real mu) {
    auto elems = tight_binding_Hamilt(n, t, mu);
    arma::mat block_ham(&elems(0, 0), n, n);
    return block_ham;
  }

  void gen_single_particle_ham() {
    sp_ham_ = arma::zeros(n_tot_, n_tot_);
    arma::mat block_11 = block_tight_binding_ham(n_left_, t_left_, mu_left_);
    arma::mat block_22 = block_tight_binding_ham(n_sys_, t_sys_, mu_sys_);
    arma::mat block_33 = block_tight_binding_ham(n_right_, t_right_, mu_right_);
    sp_ham_.submat(0, 0, n_left_ - 1, n_left_ - 1) = block_11;
    sp_ham_.submat(n_left_, n_left_, n_left_ + n_sys_ - 1, n_left_ + n_sys_ - 1) =
        block_22;
    sp_ham_.submat(n_left_ + n_sys_, n_left_ + n_sys_, n_tot_ - 1, n_tot_ - 1) =
        block_33;
    sp_ham_(n_left_ - 1, n_left_) = -t_left_sys_;
    sp_ham_(n_left_, n_left_ - 1) = -t_left_sys_;
    sp_ham_(n_left_ + n_sys_ - 1, n_left_ + n_sys_) = -t_right_sys_;
    sp_ham_(n_left_ + n_sys_, n_left_ + n_sys_ - 1) = -t_right_sys_;
  }

  void gen_auto_mpo() {
    for (int i = 0; i < n_tot_; ++i) {
      for (int j = 0; j < n_tot_; ++j) {
        double coef = hybrid_ham_(i, j);
        if (i == j) {
          ampo_ += coef, "N", i + 1;
        } else {
          ampo_ += coef, "Cdag", i + 1, "C", j + 1;
        }
      }
    }
  }

  void gen_real_space_auto_mpo() {
    for (int i = 0; i < n_tot_; ++i) {
      for (int j = 0; j < n_tot_; ++j) {
        double coef = sp_ham_(i, j);
        if (i == j) {
          rs_ampo_ += coef, "N", i + 1;
        } else {
          rs_ampo_ += coef, "Cdag", i + 1, "C", j + 1;
        }
      }
    }
  }
};

class AndersonImpurity : public MPOModel {
 public:
  AndersonImpurity(
      int left_size, int system_size, int right_size, itensor::Args const& args
  )
      : MPOModel(left_size, system_size, right_size) {}
};

class KondoImpurity : public MPOModel {
 public:
  KondoImpurity(
      int left_size, int system_size, int right_size, itensor::Args const& args
  )
      : MPOModel(left_size, system_size, right_size) {}
};

#endif  // HYBRIDLEADS_HYBRIDBASIS_MPO_MODEL_H_
