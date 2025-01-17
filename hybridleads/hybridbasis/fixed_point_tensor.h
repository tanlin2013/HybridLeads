#ifndef HYBRIDLEADS_HYBRIDBASIS_FIXED_POINT_TENSOR_H_
#define HYBRIDLEADS_HYBRIDBASIS_FIXED_POINT_TENSOR_H_

#include <glog/logging.h>

#include <map>
#include <tuple>

#include "hybridbasis/utils.h"
#include "itdvp/iTDVP.h"
#include "itensor/all.h"

using Real = itensor::Real;

enum Side { Left, Right };

class FixedPointTensor {
 public:
  /**
   * @brief Construct a new Fixed-Point Tensor object.
   *
   * @param mpo The instance of class: `itensor::MPO`.
   * @param uniform_site The site to which there are least one identical MPO
   * tensor in adjacent.
   * @param args Arguments containing the iTDVP parameters with these keywords:
   * @param time_steps (int), default 30.
   * @param dt (Real), default `INFINITY`.
   * @param tdvp_tol (Real), default 1e-12.
   * @param tdvp_max_iter (int), default 40.
   * @param max_bond_dim (int), default 1.
   * @param ortho_tol (Real), default 1e-12.
   * @param ortho_max_iter (int), , default 20.
   * @param seed (int), default 0.
   * @example
   * \code
   * Args itdvp_args = {"max_bond_dim", 10};
   * \endcode
   * @throws `std::invalid_argument` when `uniform_site` does not give any
   * identical MPO tensors as its neighbour.
   */
  FixedPointTensor(
      itensor::MPO const& mpo, int uniform_site,
      itensor::Args const& args = itensor::Args::global()
  ) {
    mpo_ = mpo;
    uniform_site_ = uniform_site;
    mpo_checker();
    int time_steps = args.getInt("time_steps", 30);
    Real dt = args.getReal("dt", INFINITY);
    int max_bond_dim = args.getInt("max_bond_dim", 1);
    Real tdvp_tol = args.getReal("tdvp_tol", 1e-12);
    int tdvp_max_iter = args.getInt("tdvp_max_iter", 40);
    Real ortho_tol = args.getReal("ortho_tol", 1e-12);
    int ortho_max_iter = args.getInt("ortho_max_iter", 20);
    RandGen::SeedType seed = args.getInt("seed", 0);
    itdvp_routine(
        time_steps, dt, max_bond_dim, tdvp_tol, tdvp_max_iter, ortho_tol,
        ortho_max_iter, seed
    );
  }

  /**
   * @brief Get the fixed-point tensor generated by iTDVP on `side`.
   *
   * @param side Enum `Side`, either `Left` or `Right`.
   * @return itensor::ITensor
   */
  itensor::ITensor get(Side side) {
    std::map<Side, int> mapper = {{Left, -1}, {Right, 1}};
    return (mapper[side] < 0) ? left_fixpt_tensor_ : right_fixpt_tensor_;
  }

  /**
   * @brief Get the shared virtual idx by fixed-point tensor on `side` and MPO.
   *
   * @param side Keyword either "Left" or "Right".
   * @return itensor::Index
   */
  itensor::Index get_mpo_virtual_idx(Side side) {
    return itensor::commonIndex(get(side), mpo_(uniform_site_));
  }

  /**
   * @brief Get the shared virtual idx by fixed-point tensor on `side` and MPS.
   *
   * @param side Keyword either "Left" or "Right".
   * @return itensor::Index
   */
  itensor::Index get_mps_virtual_idx(Side side) {
    return itensor::uniqueInds(get(side), mpo_(uniform_site_))(1);
  }

  /**
   * @brief Get the preset uniform site in constructor.
   *
   * @return int
   */
  int uniform_site() { return uniform_site_; }

  /**
   * @brief Get the unform `MPS` tensors.
   *
   * @return std::tuple<itensor::ITensor, itensor::ITensor, itensor::ITensor,
   * itensor::ITensor>
   */
  std::tuple<itensor::ITensor, itensor::ITensor, itensor::ITensor, itensor::ITensor>
  uniform_mps() {
    return {imps_left_, imps_right_, imps_center_ts_, imps_center_mat_};
  }

  /**
   * @brief Get the energy info from iTDVP routine.
   *
   * @return std::tuple<Real, Real>
   */
  std::tuple<Real, Real> get_energy_info() { return {en_, err_}; }

 protected:
  itensor::MPO mpo_;
  int uniform_site_;
  itensor::Index mpo_left_idx_, mpo_right_idx_, phys_idx_;
  itensor::ITensor imps_left_, imps_right_, imps_center_ts_, imps_center_mat_,
      left_env_, right_env_, left_fixpt_tensor_, right_fixpt_tensor_;
  Real en_, err_;

  /**
   * @brief Check at least one neighbouring MPO tensor is identical to uniform_site_.
   */
  void mpo_checker() {
    int neighbour_site = 0;
    for (int shift : {-1, 1}) {
      if (itensor::order(mpo_(uniform_site_ + shift)) ==
              itensor::order(mpo_(uniform_site_)) &&
          ALLCLOSE(mpo_(uniform_site_ + shift), mpo_(uniform_site_))) {
        neighbour_site = uniform_site_ + shift;
        break;
      }
    }
    if (neighbour_site == 0) {
      throw std::invalid_argument(
          "The `uniform site` should be picked from the bulk, with at least "
          "one neighbouring MPO tensor being identical with MPO tensor on this "
          "site."
      );
    }
  }

  /**
   * @brief Helper function for getting the virtual and physical indices.
   *
   */
  void get_indices() {
    mpo_left_idx_ = commonIndex(mpo_(uniform_site_ - 1), mpo_(uniform_site_));
    mpo_right_idx_ = commonIndex(mpo_(uniform_site_), mpo_(uniform_site_ + 1));
    phys_idx_ = findIndex(mpo_(uniform_site_), "Site,0");
  }

  /**
   * @brief Run the iTDVP algorithm to generate fixed-point tensors.
   *
   * @param time_steps
   * @param dt
   * @param max_bond_dim
   * @param tdvp_tol
   * @param tdvp_max_iter
   * @param ortho_tol
   * @param ortho_max_iter
   * @param seed
   */
  void itdvp_routine(
      int time_steps, Real dt, int max_bond_dim, Real tdvp_tol, int tdvp_max_iter,
      Real ortho_tol, int ortho_max_iter, RandGen::SeedType seed
  ) {
    get_indices();
    auto impo = mpo_(uniform_site_);
    auto imps = ITensor();  // ill-defined tensor
    std::tie(
        imps_left_, imps_right_, imps_center_ts_, imps_center_mat_, left_env_,
        right_env_
    ) =
        itdvp_initial(
            impo, phys_idx_, mpo_left_idx_, mpo_right_idx_, imps, max_bond_dim,
            ortho_tol, ortho_max_iter, seed
        );
    itensor::Args args = {"ErrGoal=", tdvp_tol, "MaxIter", tdvp_max_iter};
    for (int i = 1; i <= time_steps; i++) {
      std::tie(en_, err_, left_fixpt_tensor_, right_fixpt_tensor_) = itdvp(
          impo, imps_left_, imps_right_, imps_center_ts_, imps_center_mat_, left_env_,
          right_env_, dt, args
      );
      DLOG(INFO
      ) << std::printf("In time step %o, energy, error = %.3e, %.3e\n", i, en_, err_);
      if (args.getReal("ErrGoal") > tdvp_tol) {
        args.add("ErrGoal=", err_ * 0.1);
      }
    }
  }
};

#endif  // HYBRIDLEADS_HYBRIDBASIS_FIXED_POINT_TENSOR_H_
