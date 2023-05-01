#ifndef HYBRIDLEADS_HYBRIDBASIS_GLUON_H_
#define HYBRIDLEADS_HYBRIDBASIS_GLUON_H_

#include <armadillo>
#include <tuple>

#include "hybridbasis/fixed_point_tensor.h"
#include "itensor/all.h"

using Real = itensor::Real;
using Index = itensor::Index;
using IndexSet = itensor::IndexSet;

class Gluon {
 public:
  Gluon(
      itensor::MPO const& mpo, itensor::SiteSet const& sites, int left_lead_size,
      int right_lead_size, itensor::Args const& itdvp_args = itensor::Args::global()
  )
      : left_fxpts_(mpo, 2, itdvp_args),
        right_fxpts_(mpo, itensor::length(mpo) - 1, itdvp_args) {
    mpo_ = mpo;
    sites_ = sites;
    left_lead_size_ = left_lead_size;
    right_lead_size_ = right_lead_size;
    sys_size_ = itensor::length(sites_);
    if (sys_size_ < 2) {
      throw std::invalid_argument("System (bulk) size should be at least 2.");
    }
    if (left_lead_size_ + sys_size_ + right_lead_size_ != itensor::length(mpo_)) {
      throw std::invalid_argument("Total size does not match with the given MPO.");
    }
  }

  int sys_size() { return sys_size_; }

  itensor::MPO sys_mpo() {
    itensor::MPO sys_mpo(sites_);
    for (int site = 1; site <= sys_size_; ++site) {
      IndexSet new_site_inds = itensor::siteInds(sys_mpo, site);
      sys_mpo.ref(site) = mpo_(left_lead_size_ + site);
      sys_mpo.ref(site).replaceInds(
          itensor::siteInds(mpo_, left_lead_size_ + site), new_site_inds
      );
    }
    sys_mpo.ref(1).replaceInds(
        {itensor::leftLinkIndex(mpo_, left_lead_size_ + 1)},
        {left_fxpts_.get_mpo_virtual_idx(Left)}
    );
    sys_mpo.ref(sys_size_).replaceInds(
        {itensor::rightLinkIndex(mpo_, left_lead_size_ + sys_size_)},
        {right_fxpts_.get_mpo_virtual_idx(Right)}
    );
    return sys_mpo;
  }

  itensor::MPS random_init_state() {
    itensor::InitState state(sites_);
    for (int i : itensor::range1(sys_size_)) {
      if (i % 2 == 1)
        state.set(i, "1");
      else
        state.set(i, "0");
    }
    itensor::MPS mps = itensor::randomMPS(state);
    itensor::ITensor first_site_rand_ts = itensor::randomITensor(
        IndexSet(left_fxpts_.get_mps_virtual_idx(Left), itensor::inds(mps(1)))
    );
    itensor::ITensor last_site_rand_ts = itensor::randomITensor(
        IndexSet(itensor::inds(mps(sys_size_)), right_fxpts_.get_mps_virtual_idx(Right))
    );
    mps.set(1, first_site_rand_ts);
    mps.set(sys_size_, last_site_rand_ts);
    return mps;
  }

  /**
   * @brief Generate an initial uniform MPS from leads.
   *
   * @param take_from_side Take the uniform MPS from which side of lead.
   * @param ortho_center The orthogonal center of MPS in mixed-canonical form.
   * @return itensor::MPS
   */
  itensor::MPS uniform_state(Side take_from_side, int ortho_center) {
    itensor::ITensor imps_left, imps_right, imps_center_ts, imps_center_mat;
    if (take_from_side == Left) {
      std::tie(imps_left, imps_right, imps_center_ts, imps_center_mat) =
          left_fxpts_.uniform_mps();
    } else if (take_from_side == Right) {
      std::tie(imps_left, imps_right, imps_center_ts, imps_center_mat) =
          right_fxpts_.uniform_mps();
    }
    if (ortho_center < 1 || ortho_center > sys_size_) {
      throw std::invalid_argument(
          "The orthogonal center should be within the range of system size."
      );
    }
    itensor::MPS mps = itensor::MPS(sites_);
    Index new_virtual_left_idx = left_fxpts_.get_mps_virtual_idx(Left);
    for (int site : itensor::range1(sys_size_)) {
      Index new_phys_idx = itensor::findIndex(itensor::inds(mps(site)), "Site");
      if (site < ortho_center) {
        mps.ref(site) = imps_left;
      } else if (site == ortho_center) {
        mps.ref(site) = imps_center_ts;
      } else if (site > ortho_center) {
        mps.ref(site) = imps_right;
      }
      IndexSet old_virtual_inds = itensor::findInds(itensor::inds(mps(site)), "Link");
      Index old_phys_idx = itensor::findIndex(itensor::inds(mps(site)), "Site");
      Index new_virtual_right_idx = Index(itensor::dim(old_virtual_inds(2)), "Link");
      mps.ref(site).replaceInds(
          IndexSet(old_phys_idx, old_virtual_inds(1), old_virtual_inds(2)),
          IndexSet(new_phys_idx, new_virtual_left_idx, new_virtual_right_idx)
      );
      new_virtual_left_idx = new_virtual_right_idx;
    }
    mps.ref(1).replaceInds(
        {itensor::findInds(itensor::inds(mps(1)), "Link")(1)},
        {left_fxpts_.get_mps_virtual_idx(Left)}
    );
    mps.ref(sys_size_).replaceInds(
        {itensor::findInds(itensor::inds(mps(sys_size_)), "Link")(2)},
        {right_fxpts_.get_mps_virtual_idx(Right)}
    );
    return mps;
  }

  /**
   * @brief
   *
   * @return itensor::ITensor
   */
  itensor::ITensor left_env() { return left_fxpts_.get(Left); }

  /**
   * @brief
   *
   * @return itensor::ITensor
   */
  itensor::ITensor right_env() { return right_fxpts_.get(Right); }

  /**
   * @brief
   *
   * @param init_state
   * @param sweeps
   * @param args
   * @return std::tuple<Real, itensor::MPS>
   */
  std::tuple<Real, itensor::MPS> dmrg(
      itensor::MPS const& init_state, itensor::Sweeps const& sweeps,
      itensor::Args const& args = itensor::Args::global()
  ) {
    return itensor::dmrg(sys_mpo(), left_env(), right_env(), init_state, sweeps, args);
  }

 protected:
  itensor::MPO mpo_;
  itensor::SiteSet sites_;
  int left_lead_size_, right_lead_size_, sys_size_;
  FixedPointTensor left_fxpts_, right_fxpts_;
};

#endif  // HYBRIDLEADS_HYBRIDBASIS_GLUON_H_
