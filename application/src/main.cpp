#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numbers>
#include <random>
#include <stdio.h>
#include <vector>

#include "../../src/dissimilarities.h"
#include "../../src/init_policies.h"
#include "../../src/kmeans.h"
#include "../../src/utils.h"
#include <fdaPDE/fdapde.h>

#include <Eigen/Dense>

using std::numbers::pi;
using namespace std::chrono;
using namespace fdapde;
namespace fs = std::filesystem;

// RandomInitPolicy, ManualInitPolicy, KppPolicy
// L2Policy, L2NormalizedPolicy R1Policy, SobolevPolicy, SobolevPolicyNormalized

int main() {
  std::string output_dir = "../output/";
  std::string data_dir = "../data/";

  if (fs::exists(output_dir)) {
    fs::remove_all(output_dir);
  }
  fs::create_directory(output_dir);

  std::ofstream file_log(output_dir + "log.txt");
  if (!file_log.is_open()) {
    std::cerr << "Error opening log file." << std::endl;
    return 1;
  }
  file_log.close();
  file_log.open(output_dir + "log.txt", std::ios::app);
  if (!file_log.is_open()) {
    std::cerr << "Error reopening log file." << std::endl;
    return 1;
  }

  // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  // SET PARAMETERS
  std::vector<unsigned> k_vec = {2, 3, 4, 5};

  unsigned max_iter = 25;

  unsigned n_runs = 10;

  std::optional<unsigned> seed = std::nullopt;

  seed = std::random_device{}(); // random seed for reproducibility
  std::mt19937_64 rng(*seed);
  // seed = params.seed; // 42; // seed for random number generator

  std::vector<double> lambda_grid;
  // lambda_grid.resize(9);
  // for (int i = 0; i < lambda_grid.size(); ++i) {
  //   lambda_grid[i] = std::pow(10, -8.0 + 0.25 * i);
  // }

  // std::optional<double> lambda = std::nullopt;
  double lambda = std::pow(10, -6.25);

  // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  std::cout << "seed: " << *seed << std::endl;
  file_log << "seed: " << *seed << std::endl;
  std::cout << "k: ";
  file_log << "k :";
  bool first_flag = true;
  for (auto k : k_vec) {
    if (!first_flag) {
      std::cout << ", ";
      file_log << ", ";
    }
    std::cout << k;
    file_log << k;
  }
  std::cout << std::endl;
  file_log << std::endl;

  auto t1 = high_resolution_clock::now();
  auto t2 = high_resolution_clock::now();
  duration<double> elapsed_time = t2 - t1;

  // read nodes, cells and boudaries from csv files:
  //   Eigen::MatrixXd nodes =
  //       csv2mat<double>("../data/mesh/points.csv");
  //   Eigen::MatrixXi cells =
  //       csv2mat<int>("../data/mesh/elements.csv");
  //   Eigen::MatrixXi boundary_nodes =
  //       csv2mat<int>("../data/mesh/boundary.csv");

  // Triangulation<3, 3> D(nodes, cells, boundary_nodes);
  Triangulation<3, 3> D("../data/mesh/points.csv", "../data/mesh/elements.csv",
                        "../data/mesh/boundary.csv", 1, 1);

  std::cout << D.node(0) << std::endl;

  // 3D
  FeSpace Vh(D, P1<1>);
  TrialFunction f(Vh);
  TestFunction v(Vh);
  auto a = integral(D)(dot(grad(f), grad(v)));
  ZeroField<3> u;
  auto F = integral(D)(u * v);

  // auto R1_3d = a.assemble();
  auto mass = integral(D)(f * v);
  auto R0 = mass.assemble();

  // PARAMETERS
  // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  L2Policy dist(R0);
  KppPolicy init(dist);

  // CLUSTERING
  unsigned best_idx;
  std::map<unsigned, std::vector<int>> best_memb;
  std::map<unsigned, Eigen::MatrixXd> best_centroids;

  fs::create_directories(output_dir + "runs");
  fs::create_directories(output_dir + "wcss");

  std::string resp_file_1 = data_dir + "z_map_CONTROL.csv";
  std::string resp_file_2 = data_dir + "z_map_SCHZ.csv";
  Eigen::MatrixXd responses =
      merge_csv2mat<double>(resp_file_2, resp_file_1, 1, 1);

  // responses = responses.topRows(100).eval();

  std::string out_best_idx_file = output_dir + "best_index.csv";
  std::ofstream file_best_idx(out_best_idx_file);
  if (!file_best_idx.is_open()) {
    std::cerr << "Error opening file: " << out_best_idx_file << std::endl;
    return 1;
  }
  file_best_idx.close();

  for (auto k : k_vec) {
    std::string k_path = output_dir + "runs/k_" + std::to_string(k);
    fs::create_directories(k_path);

    std::string wcss_path = output_dir + "wcss/k_" + std::to_string(k);
    fs::create_directories(wcss_path);

    std::string out_wcss_file = wcss_path + "/wcss.csv";
    std::ofstream file_wcss(out_wcss_file);
    if (!file_wcss.is_open()) {
      std::cerr << "Error opening file: " << out_wcss_file << std::endl;
      return 1;
    }
    file_wcss.close();

    double best_wcss = std::numeric_limits<double>::max();

    for (unsigned run = 0; run < n_runs; ++run) {
      std::string run_path = k_path + "/run_" + std::to_string(run);
      fs::create_directories(run_path);

      std::string out_memb_file = run_path + "/memberships.csv";
      std::string out_cent_file = run_path + "/centroids.csv";
      std::ofstream file_memb(out_memb_file);
      std::ofstream file_cent(out_cent_file);
      if (!file_memb.is_open() || !file_cent.is_open()) {
        std::cerr << "Error opening file: " << out_memb_file << " or "
                  << out_cent_file << std::endl;
        return 1;
      }
      file_memb.close();
      file_cent.close();

      unsigned n_iter;
      std::vector<int> temp_memb;
      Eigen::MatrixXd temp_centroids;
      t1 = high_resolution_clock::now();

      RKMeans rkmeans(dist, init, D, fe_ls_elliptic(a, F), responses, k,
                      max_iter, rng());
      rkmeans.set_gcv_grid(lambda_grid);
      rkmeans.run(lambda);
      n_iter = rkmeans.n_iterations();
      temp_memb = rkmeans.memberships();
      temp_centroids = rkmeans.centroids();

      t2 = high_resolution_clock::now();
      elapsed_time = duration_cast<duration<double>>(t2 - t1);

      // Compute WCSS
      double wcss = 0.0;
      for (unsigned i = 0; i < responses.rows(); ++i) {
        wcss += std::pow(
            dist(responses.row(i), temp_centroids.row(temp_memb[i])), 2);
      }

      file_wcss.open(out_wcss_file, std::ios::app);
      if (file_wcss.is_open()) {
        file_wcss << wcss << "\n";
        file_wcss.close();
      } else {
        std::cerr << "Unable to open file" << out_wcss_file << std::endl;
        return 1;
      }

      if (wcss < best_wcss) {
        best_wcss = wcss;
        best_idx = run;
        best_memb[k] = temp_memb;
        best_centroids[k] = temp_centroids;
      }

      std::ostringstream ss;
      ss << "K = " << k << ": run " << run << " completed in " << n_iter
         << " iterations (max=" << max_iter << "), time:" << elapsed_time
         << ", WCSS: " << wcss;
      std::string msg = ss.str();
      std::cout << msg << std::endl;
      file_log << msg << std::endl;

      Eigen::Map<const Eigen::RowVectorXi> temp_row_view(temp_memb.data(),
                                                         temp_memb.size());
      append_mat2csv(temp_row_view, out_memb_file);
      append_mat2csv(temp_centroids, out_cent_file);
    }
    file_best_idx.open(out_best_idx_file, std::ios::app);
    if (file_best_idx.is_open()) {
      file_best_idx << best_idx << "\n";
      file_best_idx.close();
    } else {
      std::cerr << "Unable to open file" << out_best_idx_file << std::endl;
      return 1;
    }
  }

  // K SELECTION WITH SILUETTE SCORE

  std::string out_siluette_file = output_dir + "siluette_scores.csv";
  std::ofstream file_siluette(out_siluette_file);
  if (!file_siluette.is_open()) {
    std::cerr << "Error opening file: " << out_siluette_file << std::endl;
    return 1;
  }
  file_siluette.close();

  for (auto k : k_vec) {
    double siluette_score = 0.0;

    std::vector<std::vector<int>> clusters(k);
    for (unsigned i = 0; i < best_memb[k].size(); ++i) {
      clusters[best_memb[k][i]].push_back(i);
    }

    for (unsigned i = 0; i < best_memb[k].size(); ++i) {
      int ci = best_memb[k][i];

      double a_i = 0.0;
      int same_cluster_count = 0;
      for (int j : clusters[ci]) {
        if (i == j)
          continue;
        a_i += dist(responses.row(i), responses.row(j));
        same_cluster_count++;
      }
      if (same_cluster_count > 0) {
        a_i /= same_cluster_count;
      }

      double b_i = std::numeric_limits<double>::max();
      for (int c = 0; c < k; ++c) {
        if (c == ci)
          continue;
        if (clusters[c].empty())
          continue;
        double avg_dist = 0.0;
        for (int j : clusters[c]) {
          avg_dist += dist(responses.row(i), responses.row(j));
        }
        avg_dist /= clusters[c].size();
        b_i = std::min(b_i, avg_dist);
      }

      double s_i = (b_i - a_i) / std::max(a_i, b_i);
      siluette_score += s_i;
    }

    siluette_score /= best_memb[k].size();

    // --- Write result ---
    file_siluette.open(out_siluette_file, std::ios::app);
    if (file_siluette.is_open()) {
      file_siluette << k << "," << siluette_score << "\n";
      file_siluette.close();
    } else {
      std::cerr << "Unable to open file " << out_siluette_file << std::endl;
      return 1;
    }
  }

  file_log.close();

  return 0;
}
