#ifndef KMEANS_H
#define KMEANS_H

#include "dissimilarities.h"
#include "init_policies.h"

#include <algorithm>
#include <iostream>
#include <limits>
#include <optional>
#include <vector>

#include <Eigen/Dense>
#include <fdaPDE/fdapde.h>

using namespace fdapde;

inline constexpr unsigned MAX_KMEANS_ITERATIONS = 100;

template <typename DistancePolicy, typename InitPolicy> class KMeans {
private:
  const Eigen::MatrixXd &Y_;
  DistancePolicy dist_;
  InitPolicy init_policy_;

  std::size_t n_obs_;
  unsigned k_;
  unsigned max_iter_;
  unsigned n_iter_ = 0;

  std::vector<int> memberships_;
  Eigen::MatrixXd centroids_;
  std::vector<int> initial_clusters_;
  std::optional<unsigned> seed_; // Seed for random/kmeans++ policies

public:
  KMeans(const Eigen::MatrixXd &Y, const DistancePolicy &dist,
         const InitPolicy &init_policy, unsigned k = 3,
         unsigned max_iter = MAX_KMEANS_ITERATIONS,
         std::optional<unsigned> seed = std::nullopt)
      : Y_(Y), dist_(dist), init_policy_(init_policy), n_obs_(Y.rows()), k_(k),
        max_iter_(max_iter),
        memberships_(n_obs_, -1), // initialize memberships with -1
        centroids_(k, Y.cols()), seed_(seed) {
    if (k_ == 0 || k_ > n_obs_) {
      throw std::runtime_error("Invalid k or data size.");
    }
    centroids_.setZero();
    initial_clusters_.reserve(k_);
  }

  // Main routine
  void run() {
    // Initialize centroids with the selected policy and
    // check if init_policy_.init can be called with a seed parameter
    if constexpr (requires { init_policy_.init(Y_, centroids_, k_, seed_); }) {
      initial_clusters_ = init_policy_.init(Y_, centroids_, k_, seed_);
    } else {
      // Otherwise call it without the seed parameter (e.g., for manual policy)
      initial_clusters_ = init_policy_.init(Y_, centroids_, k_);
    }
    bool f_changed = true; // bool to check if memberships changed
    for (n_iter_ = 0; n_iter_ < max_iter_ && f_changed; ++n_iter_) {
      f_changed = false;

      // Assignment step
      for (std::size_t i = 0; i < n_obs_; ++i) {
        double best_dist = std::numeric_limits<double>::max();
        int best_c = memberships_[i];
        auto f_i = Y_.row(i);

        for (unsigned c = 0; c < k_; ++c) {
          double d = dist_(f_i, centroids_.row(c));
          if (d < best_dist) {
            best_dist = d;
            best_c = static_cast<int>(c);
          }
        }

        if (best_c != memberships_[i]) {
          memberships_[i] = best_c;
          f_changed = true;
        }
      }

      // Exit earlier, since if memberships did not change
      // => neither will the centroids, counts, etc.
      if (!f_changed) {
        break;
      }

      // Update step
      centroids_.setZero();
      std::vector<std::size_t> counts(k_, 0);

      for (std::size_t i = 0; i < n_obs_; ++i) {
        int c = memberships_[i];
        centroids_.row(c) += Y_.row(i);
        counts[c]++;
      }
      for (unsigned c = 0; c < k_; ++c) {
        if (counts[c] > 0) {
          centroids_.row(c) /= double(counts[c]);
        }
      }
    }
    /*
    std::cout << " Execution completed in " << n_iter_
              << " iterations (max=" << max_iter_ << ").\n";
    */
  }

  // Methods to extract memberships, centroids, n_iterations
  const std::vector<int> &memberships() const { return memberships_; }
  const Eigen::MatrixXd &centroids() const { return centroids_; }
  unsigned n_iterations() const { return n_iter_; }
};

// RKMeans class for regularized KMeans
template <typename DistancePolicy, typename InitPolicy, typename Triangulation,
          typename Penalty>
class RKMeans {
private:
  const Eigen::MatrixXd &Y_;
  DistancePolicy dist_;
  InitPolicy init_policy_;
  Triangulation data_;
  std::size_t n_obs_;
  Penalty penalty_;

  unsigned k_;
  unsigned max_iter_;
  unsigned n_iter_ = 0;

  std::vector<double> lambda_grid_;

  std::vector<int> memberships_;
  Eigen::MatrixXd centroids_;
  std::vector<int> initial_clusters_;
  std::optional<unsigned> seed_; // Seed for random/kmeans++ policies

  void regularize_centroids(std::optional<double> lambda = std::nullopt) {
    for (unsigned c = 0; c < k_; ++c) {
      GeoFrame data(data_);
      auto &l1 = data.template insert_scalar_layer<POINT>("obs", MESH_NODES);
      l1.load_vec("y", centroids_.row(c));

      SRPDE<typename Penalty::solver_t> model("y ~ f", data, penalty_);
      if (lambda) {
        model.fit(*lambda);
      } else {
        // calibration
        GridOptimizer<1> optimizer;
        int seed = (seed_) ? *seed_ : std::random_device{}();
        // if (c == 0) {
        //   optimizer.optimize(model.gcv(100, seed), lambda_grid_);
        //   edf_cache = model.gcv().edf_cache();
        // } else {
        //   optimizer.optimize(model.gcv(edf_cache, 100, seed), lambda_grid_);
        // }
        auto gcv = model.gcv(100, seed);
        optimizer.optimize(gcv, lambda_grid_);
        gcv.edf_cache().clear();
        std::cout << "Optimal lambda for cluster " << c << ": "
                  << optimizer.optimum()[0] << "\n";
        // if (optimizer.optimum()[0] == lambda_grid_.back() ||
        //     optimizer.optimum()[0] == lambda_grid_.front()) {
        //   std::cerr << "Warning: Optimal lambda is at the edge of the grid. "
        //             << "Consider expanding the grid for better results.\n";
        // }
        model.fit(optimizer.optimum());
      }

      centroids_.row(c) = model.fitted();
    }
  };

public:
  RKMeans(const DistancePolicy &dist, const InitPolicy &init_policy,
          const Triangulation &triang, const Penalty &penalty,
          const Eigen::MatrixXd &Y, unsigned k = 3,
          unsigned max_iter = MAX_KMEANS_ITERATIONS,
          std::optional<unsigned> seed = std::nullopt)
      : Y_(Y), dist_(dist), init_policy_(init_policy), n_obs_(Y.rows()), k_(k),
        max_iter_(max_iter), data_(triang), penalty_(penalty),
        memberships_(n_obs_, -1), // initialize memberships with -1
        centroids_(k, Y.cols()), seed_(seed) {
    if (k_ == 0 || k_ > n_obs_) {
      throw std::runtime_error("Invalid k or data size.");
    }
    lambda_grid_.resize(25);
    for (int i = 0; i < 25; ++i) {
      lambda_grid_[i] = std::pow(10, -8.0 + 0.25 * i);
    }
    centroids_.setZero();
    initial_clusters_.reserve(k_);
  }

  // Main routine
  void run(std::optional<double> lambda = std::nullopt) {
    // Initialize centroids with the selected policy and
    // check if init_policy_.init can be called with a seed parameter
    if constexpr (requires { init_policy_.init(Y_, centroids_, k_, seed_); }) {
      initial_clusters_ = init_policy_.init(Y_, centroids_, k_, seed_);
    } else {
      // Otherwise call it without the seed parameter (e.g., for manual
      // policy)
      initial_clusters_ = init_policy_.init(Y_, centroids_, k_);
    }

    // REGULARIZE CENTROIDS
    regularize_centroids(lambda);

    bool f_changed = true; // bool to check if memberships changed
    for (n_iter_ = 0; n_iter_ < max_iter_ && f_changed; ++n_iter_) {
      f_changed = false;

      // Assignment step
      for (std::size_t i = 0; i < n_obs_; ++i) {
        double best_dist = std::numeric_limits<double>::max();
        int best_c = memberships_[i];
        auto f_i = Y_.row(i);

        for (unsigned c = 0; c < k_; ++c) {
          double d = dist_(f_i, centroids_.row(c));
          if (d < best_dist) {
            best_dist = d;
            best_c = static_cast<int>(c);
          }
        }

        if (best_c != memberships_[i]) {
          memberships_[i] = best_c;
          f_changed = true;
        }
      }

      // Exit earlier, since if memberships did not change
      // => neither will the centroids, counts, etc.
      if (!f_changed) {
        break;
      }

      // Update step
      centroids_.setZero();
      std::vector<std::size_t> counts(k_, 0);

      for (std::size_t i = 0; i < n_obs_; ++i) {
        int c = memberships_[i];
        centroids_.row(c) += Y_.row(i);
        counts[c]++;
      }
      for (unsigned c = 0; c < k_; ++c) {
        if (counts[c] > 0) {
          centroids_.row(c) /= double(counts[c]);
        }
      }

      // REGULARIZE CENTROIDS
      regularize_centroids(lambda);
    }
    // std::cout << " Execution completed in " << n_iter_
    //           << " iterations (max=" << max_iter_ << ").\n";
  }

  void set_gcv_grid(std::vector<double> grid) { lambda_grid_ = grid; }

  // Methods to extract memberships, centroids, n_iterations
  const std::vector<int> &memberships() const { return memberships_; }
  const Eigen::MatrixXd &centroids() const { return centroids_; }
  unsigned n_iterations() const { return n_iter_; }
};

// RKMeans_na class for regularized KMeans MISSING DATA
template <typename DistancePolicy, typename InitPolicy, typename Triangulation,
          typename Penalty>
class RKMeans_na {
private:
  // const Eigen::MatrixXd &Y_;
  DistancePolicy dist_;
  InitPolicy init_policy_;
  Triangulation data_;
  std::size_t n_obs_;
  Penalty penalty_;

  unsigned k_;
  unsigned max_iter_;
  unsigned n_iter_ = 0;

  std::vector<double> lambda_grid_;

  Eigen::MatrixXi obs_patt_;
  double tol_; // Tolerance for convergence
  std::vector<int> memberships_;
  Eigen::MatrixXd centroids_;
  std::vector<int> initial_clusters_;
  std::optional<unsigned> seed_; // Seed for random/kmeans++ policies
  Eigen::MatrixXd Yn;

  void regularize_centroids(std::optional<double> lambda = std::nullopt) {
    for (unsigned c = 0; c < k_; ++c) {
      GeoFrame data(data_);
      auto &l1 = data.template insert_scalar_layer<POINT>("obs", MESH_NODES);
      l1.load_vec("y", centroids_.row(c));

      SRPDE<typename Penalty::solver_t> model("y ~ f", data, penalty_);
      if (lambda) {
        model.fit(*lambda);
      } else {
        // calibration
        GridOptimizer<1> optimizer;
        int seed = (seed_) ? *seed_ : std::random_device{}();
        auto gcv = model.gcv(100, seed);
        optimizer.optimize(gcv, lambda_grid_);
        gcv.edf_cache().clear();
        std::cout << "Optimal lambda for cluster " << c << ": "
                  << optimizer.optimum()[0] << "\n";
        // if (optimizer.optimum()[0] == lambda_grid_.back() ||
        //     optimizer.optimum()[0] == lambda_grid_.front()) {
        //   std::cerr << "Warning: Optimal lambda is at the edge of the grid. "
        //             << "Consider expanding the grid for better results.\n";
        // }
        model.fit(optimizer.optimum());
      }

      centroids_.row(c) = model.fitted();
    }
  };

public:
  RKMeans_na(const DistancePolicy &dist, const InitPolicy &init_policy,
             const Triangulation &triang, const Penalty &penalty,
             const Eigen::MatrixXd &Y, const Eigen::MatrixXi &obs_patt,
             double tol = 1e-6, unsigned k = 3,
             unsigned max_iter = MAX_KMEANS_ITERATIONS,
             std::optional<unsigned> seed = std::nullopt)
      : obs_patt_(obs_patt), tol_(tol), dist_(dist), init_policy_(init_policy),
        n_obs_(Y.rows()), k_(k), max_iter_(max_iter), data_(triang),
        penalty_(penalty),
        memberships_(n_obs_, -1), // initialize memberships with -1
        centroids_(k, Y.cols()), seed_(seed), Yn(Y) {
    if (k_ == 0 || k_ > n_obs_) {
      throw std::runtime_error("Invalid k or data size.");
    }
    lambda_grid_.resize(25);
    for (int i = 0; i < 25; ++i) {
      lambda_grid_[i] = std::pow(10, -8.0 + 0.25 * i);
    }
    centroids_.setZero();
    initial_clusters_.reserve(k_);

    // a. initialize Yn reconstruction with dineof
    // im assuming the data put in in this is already reconstructed using dineof
    
    // b. initialize Yn with the mean of the observed values
    Eigen::RowVectorXd mean_obs = Eigen::RowVectorXd::Zero(Y.cols());
    for (std::size_t j = 0; j < Y.cols(); ++j) {
      int count = 0;
      for (std::size_t i = 0; i < Y.rows(); ++i) {
        if (obs_patt_(i, j) == 1) {
          mean_obs(j) += Y(i, j);
          count++;
        }
      }
      if (count > 0) {
        mean_obs(j) /= count;
      } else {
        mean_obs(j) =
            j == 0 ? 0.0 : mean_obs(j - 1); // If no observations, set to 0 ,
                                            // identical to previous
      }
    }
    for (std::size_t i = 0; i < Y.rows(); ++i) {
      auto mask = obs_patt_.row(i).array().cast<bool>();
      Yn.row(i) = mask.select(Yn.row(i).array(), mean_obs.array()).matrix();
    }
    
  }

  // Main routine
  void run(std::optional<double> lambda = std::nullopt) {

    // Initialize centroids with the selected policy and
    // check if init_policy_.init can be called with a seed parameter
    if constexpr (requires { init_policy_.init(Yn, centroids_, k_, seed_); }) {
      initial_clusters_ = init_policy_.init(Yn, centroids_, k_, seed_);
    } else {
      // Otherwise call it without the seed parameter (e.g., for manual
      // policy)
      initial_clusters_ = init_policy_.init(Yn, centroids_, k_);
    }

    // REGULARIZE CENTROIDS
    regularize_centroids(lambda);

    bool f_changed = true; // bool to check if memberships changed
    for (n_iter_ = 0; n_iter_ < max_iter_ && f_changed; ++n_iter_) {
      f_changed = false;

      // Assignment step
      for (std::size_t i = 0; i < n_obs_; ++i) {
        double best_dist = std::numeric_limits<double>::max();
        int best_c = memberships_[i];

        for (unsigned c = 0; c < k_; ++c) {
          // Eigen::MatrixXd temp_centr = centroids_.row(c);
          /*
          auto mask = obs_patt_.row(i).array().cast<bool>();
          temp_centr = mask.select(temp_centr.array(),
                                   Eigen::RowVectorXd::Zero(Yn.cols()).array())
                           .matrix();
          double d = dist_(Yn.row(i), temp_centr);
          */
          double d = dist_(Yn.row(i), centroids_.row(c));
          /*
          int size_obs = obs_patt_.row(i).count();
          if (size_obs == 0) {
            std::cerr << "Warning: No observations for row " << i
                      << ". Using zero distance." << std::endl;
            d = 0;
          } else {
            d = d / size_obs;
          }
          */
          if (d < best_dist) {
            best_dist = d;
            best_c = static_cast<int>(c);
          }
        }

        if (best_c != memberships_[i]) {
          memberships_[i] = best_c;
          f_changed = true;
        }
      }

      // Exit earlier, since if memberships did not change
      // => neither will the centroids, counts, etc.
      if (!f_changed) {
        break;
      }

      // Update step
      centroids_.setZero();
      std::vector<std::size_t> counts(k_, 0);

      for (std::size_t i = 0; i < n_obs_; ++i) {
        int c = memberships_[i];
        centroids_.row(c) += Yn.row(i);
        counts[c]++;
      }
      for (unsigned c = 0; c < k_; ++c) {
        if (counts[c] > 0) {
          centroids_.row(c) /= double(counts[c]);
        }
      }

      // REGULARIZE CENTROIDS
      regularize_centroids(lambda);

      // reconstruct Yn
      for (std::size_t i = 0; i < n_obs_; ++i) {
        auto mask = obs_patt_.row(i).array().cast<bool>();
        Yn.row(i) = mask.select(Yn.row(i).array(),
                                centroids_.row(memberships_[i]).array())
                        .matrix();
      }
    }
    // std::cout << " Execution completed in " << n_iter_
    //           << " iterations (max=" << max_iter_ << ").\n";
  }

  void set_gcv_grid(std::vector<double> grid) { lambda_grid_ = grid; }

  // Methods to extract memberships, centroids, n_iterations
  const std::vector<int> &memberships() const { return memberships_; }
  const Eigen::MatrixXd &centroids() const { return centroids_; }
  unsigned n_iterations() const { return n_iter_; }
  const Eigen::MatrixXd &recontructed_input() const { return Yn; }
};

#endif // KMEANS_H
