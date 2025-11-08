// Generate meshes and responses: wave, gauss, etc.
// 1D, 2D, 3D

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numbers>
#include <random>
#include <stdio.h>
#include <vector>

#include "../src/utils.h"
#include <fdaPDE/fdapde.h>

#include <Eigen/Dense>

using std::numbers::pi;
using namespace fdapde;

int main() {

  std::string output_dir = "../";

  unsigned seed;
  seed = std::random_device{}(); // Use a random seed for generation
  // seed = 42;

  std::string combination = "021102"; // "001122", "021102"
  unsigned N = 50;
  unsigned k = 3;
  unsigned K_na = 20;        // areal missing data clusters
  double missing_perc = 0.5; // 50% missing data
  std::size_t n_obs_per_clust = 30;

  for (auto &dir : {"1d", "2d", "3d"}) { // 2.5 needs to be added
    if (std::filesystem::exists(output_dir + dir)) {
      std::filesystem::remove_all(output_dir + dir);
    }
    std::filesystem::create_directory(output_dir + dir);
  }

  Triangulation<1, 1> D1 = Triangulation<1, 1>::UnitInterval(51);
  Triangulation<2, 2> D2 = Triangulation<2, 2>::UnitSquare(31);
  Triangulation<3, 3> D3 = Triangulation<3, 3>::UnitCube(21);

  mat2csv(D1.nodes(), output_dir + "1d/nodes.csv");
  mat2csv(D2.nodes(), output_dir + "2d/nodes.csv");
  mat2csv(D3.nodes(), output_dir + "3d/nodes.csv");

  mat2csv(D1.cells(), output_dir + "1d/cells.csv");
  mat2csv(D2.cells(), output_dir + "2d/cells.csv");
  mat2csv(D3.cells(), output_dir + "3d/cells.csv");

  binmat2csv(D1.boundary_nodes(), output_dir + "1d/boundary_nodes.csv");
  binmat2csv(D2.boundary_nodes(), output_dir + "2d/boundary_nodes.csv");
  binmat2csv(D3.boundary_nodes(), output_dir + "3d/boundary_nodes.csv");

  auto nodes_1d = D1.nodes();
  auto nodes_2d = D2.nodes();
  auto nodes_3d = D3.nodes();

  auto wave_1d_0 = [](double x) {
    return std::max(1 - 10 / 3.0 * std::abs(x - 0.5), 0.0);
  };
  // auto wave_1d_0s1 = [wave_1d_0](double x) { return wave_1d_0(x - 0.2); };
  // auto wave_1d_0s2 = [wave_1d_0](double x) { return wave_1d_0(x + 0.2); };

  auto gauss_1d_0 = [](double x) {
    return std::exp(-std::pow(x - 0.5, 2) / 0.1);
  };
  // auto gauss_1d_0s1 = [gauss_1d_0](double x) { return gauss_1d_0(x - 0.2); };
  // auto gauss_1d_0s2 = [gauss_1d_0](double x) { return gauss_1d_0(x + 0.2); };

  auto wave_2d_0 = [](double x, double y) {
    double dx = x - 0.5;
    double dy = y - 0.5;
    return std::max(1 - 10 / 3.0 * std::max(std::abs(dx), std::abs(dy)), 0.0);
  };
  // auto wave_2d_0s1 = [wave_2d_0](double x, double y) {
  //   return wave_2d_0(x - 0.2, y - 0.2);
  // };
  // auto wave_2d_0s2 = [wave_2d_0](double x, double y) {
  //   return wave_2d_0(x + 0.2, y + 0.2);
  // };

  auto gauss_2d_0 = [](double x, double y) {
    return std::exp(-std::pow(x - 0.5, 2) / 0.1 - std::pow(y - 0.5, 2) / 0.1);
  };
  // auto gauss_2d_0s1 = [gauss_2d_0](double x, double y) {
  //   return gauss_2d_0(x - 0.2, y - 0.2);
  // };
  // auto gauss_2d_0s2 = [gauss_2d_0](double x, double y) {
  //   return gauss_2d_0(x + 0.2, y + 0.2);
  // };

  auto wave_3d_0 = [](double x, double y, double z) {
    double dx = x - 0.5;
    double dy = y - 0.5;
    double dz = z - 0.5;
    return std::max(
        1 - 10 / 3.0 * std ::max({std::abs(dx), std::abs(dy), std::abs(dz)}),
        0.0);
  };
  // auto wave_3d_0s1 = [wave_3d_0](double x, double y, double z) {
  //   return wave_3d_0(x - 0.2, y - 0.2, z - 0.2);
  // };
  // auto wave_3d_0s2 = [wave_3d_0](double x, double y, double z) {
  //   return wave_3d_0(x + 0.2, y + 0.2, z + 0.2);
  // };

  auto gauss_3d_0 = [](double x, double y, double z) {
    return std::exp(-std::pow(x - 0.5, 2) / 0.1 - std::pow(y - 0.5, 2) / 0.1 -
                    std::pow(z - 0.5, 2) / 0.1);
  };
  // auto gauss_3d_0s1 = [gauss_3d_0](double x, double y, double z) {
  //   return gauss_3d_0(x - 0.2, y - 0.2, z - 0.2);
  // };
  // auto gauss_3d_0s2 = [gauss_3d_0](double x, double y, double z) {
  //   return gauss_3d_0(x + 0.2, y + 0.2, z + 0.2);
  // };

  auto warped_bump_1d_0 = [](double x) {
    double d = x - 0.5;
    return std::max(
        1.0 - 8.0 * d * d + 0.2 * d * d * d + 0.1 * std::sin(20.0 * x), 0.0);
  };

  auto warped_bump_2d_0 = [](double x, double y) {
    double dx = x - 0.5;
    double dy = y - 0.5;
    double a = 0.2;
    double r2 = dx * dx + dy * dy;
    double base = 1.0 - 8.0 * r2;
    double warped = base + a * dx * dx * dx + 0.1 * std::sin(20.0 * x);

    return std::max(warped, 0.0);
  };

  auto warped_bump_3d_0 = [](double x, double y, double z) {
    double dx = x - 0.5;
    double dy = y - 0.5;
    double dz = z - 0.5;
    double a = 0.2;
    double r2 = dx * dx + dy * dy + dz * dz;
    double base = 1.0 - 8.0 * r2;
    double warped = base + a * dx * dx * dx + 0.1 * std::sin(20.0 * x);

    return std::max(warped, 0.0);
  };

  auto spline_like_1d = [](double x) {
    // Define a few control points
    std::vector<double> knots = {0.0, 0.25, 0.5, 0.75, 1.0};
    std::vector<double> values = {0.0, 0.8, 0.2, 1.0, 0.0};

    // Linear search for interval
    for (size_t i = 0; i < knots.size() - 1; ++i) {
      if (x >= knots[i] && x <= knots[i + 1]) {
        double t = (x - knots[i]) / (knots[i + 1] - knots[i]);
        double y0 = values[i];
        double y1 = values[i + 1];
        // Use smoothstep for simplicity (C1 continuous)
        double t_smooth = t * t * (3 - 2 * t);
        return y0 + (y1 - y0) * t_smooth;
      }
    }

    return 0.0;
  };

  std::size_t n_obs = n_obs_per_clust * k;
  std::mt19937 gen(seed);
  std::normal_distribution<double> noise(0, std::sqrt(0.1));
  std::uniform_real_distribution<double> unif_dist(0.0, 1.0);

  // save parameters u and seed for reproducibility on a gen log file
  std::ofstream log_out(output_dir + "gen_log.txt");
  if (!log_out.is_open()) {
    std::cerr << "Error opening gen_log.txt for writing." << std::endl;
    return 1;
  }
  log_out << "combination:     " << combination << "\n";
  log_out << "n_obs_per_clust: " << n_obs_per_clust << "\n";
  log_out << "n_clust:         " << k << "\n";
  log_out << "N:               " << N << "\n";
  log_out << "K_na:            " << K_na << "\n";
  log_out << "missing_perc:    " << missing_perc << "\n";
  log_out << "seed:            " << seed << "\n";
  log_out << "u: [\n";
  log_out.close();

  for (auto &dir : {"1d", "2d", "3d"}) {
    for (auto &dir1 : {"wave", "gauss", "warped_bump", "spline_like"}) {
      if (std::filesystem::exists(output_dir + dir + "/" + dir1)) {
        std::filesystem::remove_all(output_dir + dir + "/" + dir1);
      }
      std::filesystem::create_directory(output_dir + dir + "/" + dir1);
    }
  }

  for (auto &dir : {"1d", "2d", "3d"}) {
    if (std::filesystem::exists(output_dir + dir + "/no_noise")) {
      std::filesystem::remove_all(output_dir + dir + "/no_noise");
    }
    std::filesystem::create_directory(output_dir + dir + "/no_noise");
  }

  for (auto &dir : {"1d", "2d", "3d"}) {
    for (auto &dir1 : {"wave", "gauss", "warped_bump", "spline_like"}) {
      if (std::filesystem::exists(output_dir + dir + "/no_noise/" + dir1)) {
        std::filesystem::remove_all(output_dir + dir + "/no_noise/" + dir1);
      }
      std::filesystem::create_directory(output_dir + dir + "/no_noise/" + dir1);
    }
  }

  // save the u onto the gen log file
  log_out.open(output_dir + "gen_log.txt", std::ios::app);
  if (!log_out.is_open()) {
    std::cerr << "Error opening gen_log.txt." << std::endl;
    return 1;
  }

  for (unsigned n = 0; n < N; ++n) {
    double u = unif_dist(gen);

    log_out << u << "\n";

    using Func1D = std::function<double(double)>;
    std::vector<std::pair<Func1D, std::string>> funcs_1d = {
        {gauss_1d_0, "gauss"},
        {wave_1d_0, "wave"},
        {warped_bump_1d_0, "warped_bump"},
        {spline_like_1d, "spline_like"},
    };

    for (auto &[base_func, wave_type] : funcs_1d) {
      auto fw_func = [base_func](double x) { return base_func(x + 0.1); };
      auto bw_func = [base_func](double x) { return base_func(x - 0.1); };
      Eigen::MatrixXd out_no_noise =
          Eigen::MatrixXd::Zero(n_obs, nodes_1d.rows());
      Eigen::MatrixXd out = Eigen::MatrixXd::Zero(n_obs, nodes_1d.rows());
      for (std::size_t i = 0; i < n_obs_per_clust; ++i) {
        for (std::size_t j = 0; j < nodes_1d.rows(); ++j) {
          double x = nodes_1d(j, 0);
          if (combination == "001122") {
            out_no_noise(i, j) = u * base_func(x) + (1 - u) * bw_func(x);
            out_no_noise(i + n_obs_per_clust, j) =
                u * base_func(x) + (1 - u) * fw_func(x);
            out_no_noise(i + 2 * n_obs_per_clust, j) =
                u * bw_func(x) + (1 - u) * fw_func(x);
          }
          if (combination == "021102") {
            out_no_noise(i, j) = u * base_func(x) + (1 - u) * bw_func(x);
            out_no_noise(i + n_obs_per_clust, j) =
                u * fw_func(x) + (1 - u) * base_func(x);
            out_no_noise(i + 2 * n_obs_per_clust, j) =
                u * bw_func(x) + (1 - u) * fw_func(x);
          }
          out(i, j) = out_no_noise(i, j) + noise(gen);
          out(i + n_obs_per_clust, j) =
              out_no_noise(i + n_obs_per_clust, j) + noise(gen);
          out(i + 2 * n_obs_per_clust, j) =
              out_no_noise(i + 2 * n_obs_per_clust, j) + noise(gen);
        }
      }
      mat2csv(out_no_noise, output_dir + "1d/no_noise/" + wave_type + "/" +
                                wave_type + "_" + std::to_string(n) + ".csv");
      mat2csv(out, output_dir + "1d/" + wave_type + "/" + wave_type + "_" +
                       std::to_string(n) + ".csv");
    }

    using Func2D = std::function<double(double, double)>;
    std::vector<std::pair<Func2D, std::string>> funcs_2d = {
        {gauss_2d_0, "gauss"},
        {wave_2d_0, "wave"},
        {warped_bump_2d_0, "warped_bump"},
    };

    for (auto &[base_func, wave_type] : funcs_2d) {
      auto fw_func = [base_func](double x, double y) {
        return base_func(x + 0.1, y);
      };
      auto bw_func = [base_func](double x, double y) {
        return base_func(x - 0.1, y);
      };
      Eigen::MatrixXd out_no_noise =
          Eigen::MatrixXd::Zero(n_obs, nodes_2d.rows());
      Eigen::MatrixXd out = Eigen::MatrixXd::Zero(n_obs, nodes_2d.rows());
      for (std::size_t i = 0; i < n_obs_per_clust; ++i) {
        for (std::size_t j = 0; j < nodes_2d.rows(); ++j) {
          double x = nodes_2d(j, 0);
          double y = nodes_2d(j, 1);
          if (combination == "001122") {
            out_no_noise(i, j) = u * base_func(x, y) + (1 - u) * bw_func(x, y);
            out_no_noise(i + n_obs_per_clust, j) =
                u * base_func(x, y) + (1 - u) * fw_func(x, y);
            out_no_noise(i + 2 * n_obs_per_clust, j) =
                u * bw_func(x, y) + (1 - u) * fw_func(x, y);
          }
          if (combination == "021102") {
            out_no_noise(i, j) = u * base_func(x, y) + (1 - u) * bw_func(x, y);
            out_no_noise(i + n_obs_per_clust, j) =
                u * fw_func(x, y) + (1 - u) * base_func(x, y);
            out_no_noise(i + 2 * n_obs_per_clust, j) =
                u * bw_func(x, y) + (1 - u) * fw_func(x, y);
          }
          out(i, j) = out_no_noise(i, j) + noise(gen);
          out(i + n_obs_per_clust, j) =
              out_no_noise(i + n_obs_per_clust, j) + noise(gen);
          out(i + 2 * n_obs_per_clust, j) =
              out_no_noise(i + 2 * n_obs_per_clust, j) + noise(gen);
        }
      }
      mat2csv(out_no_noise, output_dir + "2d/no_noise/" + wave_type + "/" +
                                wave_type + "_" + std::to_string(n) + ".csv");
      mat2csv(out, output_dir + "2d/" + wave_type + "/" + wave_type + "_" +
                       std::to_string(n) + ".csv");
    }

    using Func3D = std::function<double(double, double, double)>;
    std::vector<std::pair<Func3D, std::string>> funcs_3d = {
        {gauss_3d_0, "gauss"},
        {wave_3d_0, "wave"},
        {warped_bump_3d_0, "warped_bump"},
    };

    for (auto &[base_func, wave_type] : funcs_3d) {
      auto fw_func = [base_func](double x, double y, double z) {
        return base_func(x + 0.1, y, z);
      };
      auto bw_func = [base_func](double x, double y, double z) {
        return base_func(x - 0.1, y, z);
      };
      Eigen::MatrixXd out_no_noise =
          Eigen::MatrixXd::Zero(n_obs, nodes_3d.rows());
      Eigen::MatrixXd out = Eigen::MatrixXd::Zero(n_obs, nodes_3d.rows());
      for (std::size_t i = 0; i < n_obs_per_clust; ++i) {
        for (std::size_t j = 0; j < nodes_3d.rows(); ++j) {
          double x = nodes_3d(j, 0);
          double y = nodes_3d(j, 1);
          double z = nodes_3d(j, 2);
          if (combination == "001122") {
            out_no_noise(i, j) = u * base_func(x, y, z) +
                                 (1 - u) * bw_func(x, y, z) + noise(gen);
            out_no_noise(i + n_obs_per_clust, j) = u * base_func(x, y, z) +
                                                   (1 - u) * fw_func(x, y, z) +
                                                   noise(gen);
            out_no_noise(i + 2 * n_obs_per_clust, j) =
                u * bw_func(x, y, z) + (1 - u) * fw_func(x, y, z);
          }
          if (combination == "021102") {
            out_no_noise(i, j) = u * base_func(x, y, z) +
                                 (1 - u) * bw_func(x, y, z) + noise(gen);
            out_no_noise(i + n_obs_per_clust, j) =
                u * fw_func(x, y, z) + (1 - u) * base_func(x, y, z) +
                noise(gen);
            out_no_noise(i + 2 * n_obs_per_clust, j) =
                u * bw_func(x, y, z) + (1 - u) * fw_func(x, y, z);
          }
          out(i, j) = out_no_noise(i, j) + noise(gen);
          out(i + n_obs_per_clust, j) =
              out_no_noise(i + n_obs_per_clust, j) + noise(gen);
          out(i + 2 * n_obs_per_clust, j) =
              out_no_noise(i + 2 * n_obs_per_clust, j) + noise(gen);
        }
      }
      mat2csv(out_no_noise, output_dir + "3d/no_noise/" + wave_type + "/" +
                                wave_type + "_" + std::to_string(n) + ".csv");
      mat2csv(out, output_dir + "3d/" + wave_type + "/" + wave_type + "_" +
                       std::to_string(n) + ".csv");
    }

    std::cout << "Generated data for iteration " << n << std::endl;
  }

  log_out << "]\n";
  log_out.close();

  for (auto &dir : {"1d", "2d", "3d"}) {
    if (std::filesystem::exists(output_dir + dir + "/observation_patterns")) {
      std::filesystem::remove_all(output_dir + dir + "/observation_patterns");
    }
    std::filesystem::create_directory(output_dir + dir +
                                      "/observation_patterns");
    for (auto &dir1 : {"scatter", "area"}) {
      if (std::filesystem::exists(output_dir + dir + "/observation_patterns/" +
                                  dir1)) {
        std::filesystem::remove_all(output_dir + dir +
                                    "/observation_patterns/" + dir1);
      }
      std::filesystem::create_directory(output_dir + dir +
                                        "/observation_patterns/" + dir1);
    }
  }

  for (auto &dir : {"1d", "2d", "3d"}) {
    Eigen::MatrixXd nodes;
    unsigned n_missing;
    unsigned n_nodes = 0;
    if (dir == "1d") {
      nodes = nodes_1d;
    }
    if (dir == "2d") {
      nodes = nodes_2d;
    }
    if (dir == "3d") {
      nodes = nodes_3d;
    }
    n_nodes = static_cast<unsigned>(nodes.rows());
    n_missing = static_cast<unsigned>(n_nodes * missing_perc);
    for (auto &dir1 : {"scatter", "area"}) {
      for (unsigned n = 0; n < N; ++n) {
        Eigen::MatrixXi observation_pattern =
            Eigen::MatrixXi::Ones(n_obs, n_nodes);
        if (dir1 == "scatter") {
          for (unsigned j = 0; j < n_obs; ++j) {
            // Randomly select nodes to be missing
            std::vector<unsigned> indices(n_nodes);
            std::iota(indices.begin(), indices.end(), 0);
            std::shuffle(indices.begin(), indices.end(), gen);
            for (unsigned i = 0; i < n_missing; ++i) {
              unsigned idx = indices[i];
              observation_pattern(j, idx) = 0;
            }
          }
          std::cout << dir << "/" << dir1
                    << ": Generated observation pattern for iteration " << n
                    << std::endl;
        }
        if (dir1 == "area") {
          for (unsigned j = 0; j < n_obs; ++j) {
            // run k means to select nodes to be missing
            std::vector<int> memberships(n_nodes, -1);
            Eigen::MatrixXd centroids(K_na, nodes.cols());

            // Initialize centroids randomly
            std::vector<int> initial_clusters;
            initial_clusters.reserve(K_na);
            std::uniform_int_distribution<int> distr(0, n_nodes - 1);
            for (unsigned i = 0; i < K_na; ++i) {
              int idx = distr(gen);
              initial_clusters.push_back(idx);
              centroids.row(i) = nodes.row(idx);
            }

            // one shot strategy (only one iteration, i dont need convergence)
            for (unsigned i = 0; i < n_nodes; ++i) {
              int best_k = 0;
              double min_dist = (nodes.row(i) - centroids.row(0)).squaredNorm();
              for (int k = 1; k < K_na; ++k) {
                double dist = (nodes.row(i) - centroids.row(k)).squaredNorm();
                if (dist < min_dist) {
                  min_dist = dist;
                  best_k = k;
                }
              }
              memberships[i] = best_k;
            }

            // select 0.5*K_na clusters to be missing
            unsigned n_missing_clusters =
                static_cast<unsigned>(K_na * missing_perc);
            std::vector<int> cluster_indices(K_na);
            std::iota(cluster_indices.begin(), cluster_indices.end(), 0);
            std::shuffle(cluster_indices.begin(), cluster_indices.end(), gen);
            for (unsigned i = 0; i < n_missing_clusters; ++i) {
              int cluster_idx = cluster_indices[i];
              for (unsigned idx = 0; idx < n_nodes; ++idx) {
                if (memberships[idx] == cluster_idx) {
                  observation_pattern(j, idx) = 0; // mark node as missing
                }
              }
            }
          }
          std::cout << dir << "/" << dir1
                    << ": Generated observation pattern for iteration " << n
                    << std::endl;
        }
        // Save the observation pattern
        mat2csv(observation_pattern,
                output_dir + dir + "/observation_patterns/" + dir1 + "/" +
                    dir1 + "_" + std::to_string(n) + ".csv");
      }
    }
  }

  std::cout << "Data generation completed successfully." << std::endl;

  return 0;
}
