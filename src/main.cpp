#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numbers>
#include <random>
#include <stdio.h>
#include <vector>

#include "dissimilarities.h"
#include "init_policies.h"
#include "kmeans.h"
#include "utils.h"
#include <fdaPDE/fdapde.h>

#include <Eigen/Dense>

using std::numbers::pi;
using namespace std::chrono;
using namespace fdapde;
namespace fs = std::filesystem;

// RandomInitPolicy, ManualInitPolicy, KppPolicy
// L2Policy, L2NormalizedPolicy R1Policy, SobolevPolicy, SobolevPolicyNormalized

int main() {
  std::vector<std::string> curve_types = {
      "gauss"}; //"wave", , "warped_bump", "spline_like"

  std::string output_dir = "../output/";
  // std::string data_dir = "../data/";

  if (fs::exists(output_dir)) {
    // fs::remove_all(output_dir);
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

  // read parameters from generation log:
  LogParams params;
  try {
    params = parse_log_scalars("../data/gen_log.txt");
    std::cout << "n_obs_per_clust: " << params.n_obs_per_clust << "\n";
    file_log << "n_obs_per_clust: " << params.n_obs_per_clust << "\n";
    std::cout << "n_clust:         " << params.n_clust << "\n";
    file_log << "n_clust:         " << params.n_clust << "\n";
    std::cout << "N:               " << params.N << "\n";
    file_log << "N:               " << params.N << "\n";
  } catch (const std::exception &e) {
    std::cerr << "Error parsing log: " << e.what() << "\n";
    return 1;
  }

  // SET PARAMETERS
  unsigned N = params.N;       // 50; // number of iterations
  unsigned k = params.n_clust; // 3;
  std::size_t n_obs_per_clust = params.n_obs_per_clust; // 10;

  // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  unsigned max_iter = 25;

  std::optional<unsigned> seed = std::nullopt;
  seed = std::random_device{}(); // random seed for reproducibility
  // seed = params.seed; // 42; // seed for random number generator

  std::optional<double> lambda = std::nullopt;
  // lambda = 1e-4; // regularization parameter for RKMeans

  double tol = 1e-4; // tolerance for convergence in RKMeans_na

  std::vector<double> lambda_1d;
  std::vector<double> lambda_2d;
  std::vector<double> lambda_3d;

  lambda_1d.resize(21);
  for (std::size_t i = 0; i < lambda_1d.size(); ++i)
    lambda_1d[i] = std::pow(10, -8 + i * 0.25); 

  lambda_2d.resize(17);
  for (std::size_t i = 0; i < lambda_2d.size(); ++i)
    lambda_2d[i] = std::pow(10, -7 + i * 0.25);

  lambda_3d.resize(13);
  for (std::size_t i = 0; i < lambda_3d.size(); ++i)
    lambda_3d[i] = std::pow(10, -6 + i * 0.25);

  // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  if (seed.has_value()) {
    std::cout << "seed:            " << *seed << "\n";
    file_log << "seed:            " << *seed << "\n";
  } else {
    std::cout << "seed:            undefined\n";
    file_log << "seed:            undefined\n";
  }

  // create output directories

  for (std::string_view dir : {"reg", "non_reg"}) {

    if (fs::exists(fs::path(output_dir) / dir)) {
      fs::remove_all(fs::path(output_dir) / dir);
    }

    for (std::string_view dir1 : {"1d", "2d", "3d"})
      for (std::string_view dir2 : curve_types)
        fs::create_directories(fs::path(output_dir) / dir / dir1 / dir2);
  }

  if (fs::exists(output_dir + "missing")) {
    fs::remove_all(output_dir + "missing");
  }
  fs::create_directories(output_dir + "missing");
  for (std::string_view dir : {"1d", "2d", "3d"})
    for (std::string_view dir1 : {"scatter", "area"})
      for (std::string_view dir2 : curve_types)
        fs::create_directories(fs::path(output_dir) / fs::path("missing") /
                               dir / dir1 / dir2);

  auto t1 = high_resolution_clock::now();
  auto t2 = high_resolution_clock::now();
  duration<double> elapsed_time = t2 - t1;

  // read nodes, cells and boudaries from csv files:
  Eigen::MatrixXd nodes_1d = csv2mat<double>("../data/1d/nodes.csv");
  Eigen::MatrixXd nodes_2d = csv2mat<double>("../data/2d/nodes.csv");
  Eigen::MatrixXd nodes_3d = csv2mat<double>("../data/3d/nodes.csv");

  Eigen::MatrixXi cells_1d = csv2mat<int>("../data/1d/cells.csv");
  Eigen::MatrixXi cells_2d = csv2mat<int>("../data/2d/cells.csv");
  Eigen::MatrixXi cells_3d = csv2mat<int>("../data/3d/cells.csv");

  Eigen::MatrixXi boundary_nodes_1d =
      csv2mat<int>("../data/1d/boundary_nodes.csv");
  Eigen::MatrixXi boundary_nodes_2d =
      csv2mat<int>("../data/2d/boundary_nodes.csv");
  Eigen::MatrixXi boundary_nodes_3d =
      csv2mat<int>("../data/3d/boundary_nodes.csv");

  Triangulation<1, 1> D1(nodes_1d);
  Triangulation<2, 2> D2(nodes_2d, cells_2d, boundary_nodes_2d);
  Triangulation<3, 3> D3(nodes_3d, cells_3d, boundary_nodes_3d);

  // PHYSICS
  // 1D
  FeSpace Vh_1d(D1, P1<1>);
  TrialFunction f_1d(Vh_1d);
  TestFunction v_1d(Vh_1d);
  auto a_1d = integral(D1)(dot(grad(f_1d), grad(v_1d)));
  ZeroField<1> u_1d;
  auto F_1d = integral(D1)(u_1d * v_1d);

  auto R1_1d = a_1d.assemble();
  auto mass_1d = integral(D1)(f_1d * v_1d);
  auto R0_1d = mass_1d.assemble();

  // 2D
  FeSpace Vh_2d(D2, P1<1>);
  TrialFunction f_2d(Vh_2d);
  TestFunction v_2d(Vh_2d);
  auto a_2d = integral(D2)(dot(grad(f_2d), grad(v_2d)));
  ZeroField<2> u_2d;
  auto F_2d = integral(D2)(u_2d * v_2d);

  auto R1_2d = a_2d.assemble();
  auto mass_2d = integral(D2)(f_2d * v_2d);
  auto R0_2d = mass_2d.assemble();

  // 3D
  FeSpace Vh_3d(D3, P1<1>);
  TrialFunction f_3d(Vh_3d);
  TestFunction v_3d(Vh_3d);
  auto a_3d = integral(D3)(dot(grad(f_3d), grad(v_3d)));
  ZeroField<3> u_3d;
  auto F_3d = integral(D3)(u_3d * v_3d);

  auto R1_3d = a_3d.assemble();
  auto mass_3d = integral(D3)(f_3d * v_3d);
  auto R0_3d = mass_3d.assemble();

  // PARAMETERS
  // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  L2Policy dist_1d(R0_1d);
  L2Policy dist_2d(R0_2d);
  L2Policy dist_3d(R0_3d);
  // KppPolicy init_1d(dist_1d);
  // KppPolicy init_2d(dist_2d);
  // KppPolicy init_3d(dist_3d);
  std::vector<int> manual_ids;
  for (std::size_t i = 0; i < k; ++i) {
    manual_ids.push_back(static_cast<int>(i * n_obs_per_clust));
  }
  ManualInitPolicy init_manual(manual_ids);

  std::size_t n_obs = n_obs_per_clust * k;

  // CLUSTERING

  for (auto &dir : {"reg", "non_reg"}) {
    for (auto &dir1 : {"1d", "2d", "3d"}) {
      unsigned n_nodes = (dir1 == "1d")   ? nodes_1d.rows()
                         : (dir1 == "2d") ? nodes_2d.rows()
                                          : nodes_3d.rows();
      for (auto &dir2 : curve_types) {
        std::string out_memb_file =
            output_dir + dir + "/" + dir1 + "/" + dir2 + "/memberships.csv";
        std::string out_cent_file =
            output_dir + dir + "/" + dir1 + "/" + dir2 + "/centroids.csv";
        std::ofstream file_memb(out_memb_file);
        std::ofstream file_cent(out_cent_file);
        if (!file_memb.is_open() || !file_cent.is_open()) {
          std::cerr << "Error opening file: " << out_memb_file << " or "
                    << out_cent_file << std::endl;
          return 1;
        }
        file_memb.close();
        file_cent.close();
        for (unsigned n = 0; n < N; ++n) {
          std::string resp_file = std::string("../data/") + dir1 + "/" + dir2 +
                                  "/" + dir2 + "_" + std::to_string(n) + ".csv";
          Eigen::MatrixXd responses = csv2mat<double>(resp_file);

          t1 = high_resolution_clock::now();
          unsigned n_iter;
          std::vector<int> temp_memb;
          Eigen::MatrixXd temp_centroids;

          if (dir == "reg") {
            if (dir1 == "1d") {
              RKMeans rkmeans(dist_1d, init_manual, D1,
                              fe_ls_elliptic(a_1d, F_1d), responses, k,
                              max_iter, seed);
              rkmeans.set_gcv_grid(lambda_1d);
              rkmeans.run(lambda);
              n_iter = rkmeans.n_iterations();
              temp_memb = rkmeans.memberships();
              temp_centroids = rkmeans.centroids();
            }
            if (dir1 == "2d") {
              RKMeans rkmeans(dist_2d, init_manual, D2,
                              fe_ls_elliptic(a_2d, F_2d), responses, k,
                              max_iter, seed);
              rkmeans.set_gcv_grid(lambda_2d);
              rkmeans.run(lambda);
              n_iter = rkmeans.n_iterations();
              temp_memb = rkmeans.memberships();
              temp_centroids = rkmeans.centroids();
            }
            if (dir1 == "3d") {
              RKMeans rkmeans(dist_3d, init_manual, D3,
                              fe_ls_elliptic(a_3d, F_3d), responses, k,
                              max_iter, seed);
              rkmeans.set_gcv_grid(lambda_3d);
              rkmeans.run(lambda);
              n_iter = rkmeans.n_iterations();
              temp_memb = rkmeans.memberships();
              temp_centroids = rkmeans.centroids();
            }
          }
          if (dir == "non_reg") {
            if (dir1 == "1d") {
              KMeans kmeans(responses, dist_1d, init_manual, k, max_iter, seed);
              kmeans.run();
              n_iter = kmeans.n_iterations();
              temp_memb = kmeans.memberships();
              temp_centroids = kmeans.centroids();
            }
            if (dir1 == "2d") {
              KMeans kmeans(responses, dist_2d, init_manual, k, max_iter, seed);
              kmeans.run();
              n_iter = kmeans.n_iterations();
              temp_memb = kmeans.memberships();
              temp_centroids = kmeans.centroids();
            }
            if (dir1 == "3d") {
              KMeans kmeans(responses, dist_3d, init_manual, k, max_iter, seed);
              kmeans.run();
              n_iter = kmeans.n_iterations();
              temp_memb = kmeans.memberships();
              temp_centroids = kmeans.centroids();
            }
          }

          t2 = high_resolution_clock::now();
          elapsed_time = duration_cast<duration<double>>(t2 - t1);

          std::string kmeans_type = (dir == "reg") ? "rkmeans" : "kmeans";
          std::ostringstream ss;
          ss << dir << "/" << dir1 << "/" << dir2 << "_" << n << ": "
             << kmeans_type << " execution completed in " << n_iter
             << " iterations (max=" << max_iter << "), time:" << elapsed_time;
          std::string msg = ss.str();
          std::cout << msg << std::endl;
          file_log << msg << std::endl;

          Eigen::Map<const Eigen::RowVectorXi> temp_row_view(temp_memb.data(),
                                                             temp_memb.size());
          append_mat2csv(temp_row_view, out_memb_file);
          append_mat2csv(temp_centroids, out_cent_file);
        }
      }
    }
  }

  for (auto &dir : {"1d", "2d", "3d"}) {
    for (auto &dir1 : {"scatter", "area"}) {
      for (auto &dir2 : curve_types) {
        std::string out_memb_file = output_dir + "missing/" + dir + "/" + dir1 +
                                    "/" + dir2 + "/memberships.csv";
        std::string out_cent_file = output_dir + "missing/" + dir + "/" + dir1 +
                                    "/" + dir2 + "/centroids.csv";
        std::ofstream file_memb(out_memb_file);
        std::ofstream file_cent(out_cent_file);
        if (!file_memb.is_open() || !file_cent.is_open()) {
          std::cerr << "Error opening file: " << out_memb_file << " or "
                    << out_cent_file << std::endl;
          return 1;
        }
        file_memb.close();
        file_cent.close();
        for (unsigned n = 0; n < N; ++n) {
          Eigen::MatrixXd responses = csv2mat<double>(
              std::string("../data/") + dir + "/" + dir2 + "/" + dir2 +
              "_" + std::to_string(n) + ".csv");
          // Eigen::MatrixXd responses = csv2mat<double>(
          //     std::string("../data/") + dir + "/no_noise/" + dir2 + "/" + dir2 +
          //     "_" + std::to_string(n) + ".csv");
          // Eigen::MatrixXd responses = csv2mat<double>(
          //     std::string("../data_reconstructed/") + dir + "/" + dir1 + "/"
          //     + dir2 + "/" + "dineof/dineof_" + std::to_string(n) + ".csv");
          Eigen::MatrixXi observation_pattern = csv2mat<int>(
              std::string("../data/") + dir + "/observation_patterns/" + dir1 +
              "/" + dir1 + "_" + std::to_string(n) + ".csv");

          t1 = high_resolution_clock::now();
          unsigned n_iter;
          std::vector<int> temp_memb;
          Eigen::MatrixXd temp_centroids;
          if (dir == "1d") {
            auto rkmeans = RKMeans_na(
                dist_1d, init_manual, D1, fe_ls_elliptic(a_1d, F_1d), responses,
                observation_pattern, tol, k, max_iter, seed);
              rkmeans.set_gcv_grid(lambda_1d);
            rkmeans.run(lambda);
            n_iter = rkmeans.n_iterations();
            temp_memb = rkmeans.memberships();
            temp_centroids = rkmeans.centroids();
          }
          if (dir == "2d") {
            auto rkmeans = RKMeans_na(
                dist_2d, init_manual, D2, fe_ls_elliptic(a_2d, F_2d), responses,
                observation_pattern, tol, k, max_iter, seed);
              rkmeans.set_gcv_grid(lambda_2d);
            rkmeans.run(lambda);
            n_iter = rkmeans.n_iterations();
            temp_memb = rkmeans.memberships();
            temp_centroids = rkmeans.centroids();
          }
          if (dir == "3d") {
            auto rkmeans = RKMeans_na(
                dist_3d, init_manual, D3, fe_ls_elliptic(a_3d, F_3d), responses,
                observation_pattern, tol, k, max_iter, seed);
              rkmeans.set_gcv_grid(lambda_3d);
            rkmeans.run(lambda);
            n_iter = rkmeans.n_iterations();
            temp_memb = rkmeans.memberships();
            temp_centroids = rkmeans.centroids();
          }

          t2 = high_resolution_clock::now();
          elapsed_time = duration_cast<duration<double>>(t2 - t1);

          std::ostringstream ss;
          ss << "missing/" << dir << "/" << dir1 << "/" << dir2 << "_" << n
             << ": rkmeans_na execution completed in " << n_iter
             << " iterations (max=" << max_iter << "), time:" << elapsed_time;
          std::string msg = ss.str();
          std::cout << msg << std::endl;
          file_log << msg << std::endl;

          Eigen::Map<const Eigen::RowVectorXi> temp_row_view(temp_memb.data(),
                                                             temp_memb.size());
          append_mat2csv(temp_row_view, out_memb_file);
          append_mat2csv(temp_centroids, out_cent_file);
        }
      }
    }
  }

  file_log.close();

  return 0;
}
