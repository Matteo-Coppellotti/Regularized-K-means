library(funFEM)
library(fda.usc)
library(funData)
library(MFPCA)
library(rTensor)
library(fdacluster)

# --- Parameters ---
output_dir <- "./output_competitors/reconstructed"
recon_dir <- "./data_reconstructed"

log_file <- "./data/gen_log.txt"
use_seed <- FALSE  # Set to FALSE to ignore seed from log

# --- Function to extract values from the log file ---
extract_log_params <- function(file_path) {
  lines <- readLines(file_path)
  extract_value <- function(pattern) {
    line <- grep(pattern, lines, value = TRUE)[1]  # only take the first match
    as.numeric(gsub(paste0(".*", pattern, "\\s*:?\\s*"), "", line))
  }
  
  list(
    n_obs_per_clust = extract_value("n_obs_per_clust"),
    n_clust         = extract_value("n_clust"),
    N               = extract_value("N"),
    seed            = extract_value("seed")
  )
}

# --- Extract parameters ---
params <- extract_log_params(log_file)

# --- Optional seeding ---
if (use_seed) {
  set.seed(params$seed)
  cat("Using seed:", params$seed, "\n")
} else {
  cat("Running unseeded (random seed)\n")
}

# --- Print extracted values ---
cat("n_obs_per_clust:", params$n_obs_per_clust, "\n")
cat("n_clust:        ", params$n_clust, "\n")
cat("N:              ", params$N, "\n")

init_centers <- seq(1,
                    by = params$n_obs_per_clust,
                    length.out = params$n_clust)

methods <- list()

for (dim in c("1d", "2d", "3d")) {
  for (type in c("scatter", "area")) {
    for (curve in c("wave", "gauss")) {
      temp_path <- file.path(recon_dir, dim, type, curve)
      if (!dir.exists(temp_path)) {
        cat(sprintf("Directory %s does not exist, skipping...\n", temp_path))
        next
      }
      if (is.null(methods[[dim]])) {
        methods[[dim]] <- character()
      }
      method_list <- list.dirs(temp_path, full.names = FALSE, recursive = FALSE)
      methods[[dim]] <- unique(c(methods[[dim]], method_list))
    }
  }
}

clustering_methods <- list()
clustering_methods[["1d"]] <- c("mv", "funFEM", "fda_usc", "fdacluster")
clustering_methods[["2d"]] <- c("mv", "tensor_pca")
clustering_methods[["3d"]] <- c("mv", "tensor_pca")

distributions <- c("gauss") #"wave", , "warped_bump", "spline_like"

# ---- MAIN LOOP 1D ----
for (space in c("1d", "2d", "3d")) { #
  nodes <- as.matrix(read.csv(file.path("data", space, "nodes.csv"), header = FALSE))
  for (recon in methods[[space]]) {
    for (method in clustering_methods[[space]]) {
      for (type in c("scatter", "area")) {
        for (dist in distributions) {
          dir_path <- file.path(output_dir, space, recon, type, method, dist)
          if (dir.exists(dir_path)) {
            unlink(dir_path, recursive = TRUE, force = TRUE)
          }
          dir.create(dir_path,
                     recursive = TRUE,
                     showWarnings = FALSE)
          
          memb_path <- file.path(dir_path, "memberships.csv")
          file.create(memb_path)
          
          for (n in 0:(params$N - 1)) {
            # Read data
            data_file <- file.path(recon_dir,
                                   space,
                                   type,
                                   dist,
                                   recon,
                                   paste0(recon, "_", n, ".csv"))
            if (!file.exists(data_file)) {
              cat(sprintf("File %s does not exist, skipping...\n", data_file))
              next
            }
            data <- as.matrix(read.csv(data_file, header = FALSE))
                      
            start_time <- Sys.time()
            # ---- Apply Clustering Method ----
            if (space == "1d") {
              result <- tryCatch({
                switch(
                  method,
                  mv = {
                    km <- kmeans(data, centers = data[init_centers, ])
                    list(cluster = km$cluster,
                         iter = km$iter)
                  },
                  funFEM = {
                    nodes_ff <- as.numeric(nodes)
                    basis <- create.bspline.basis(rangeval = range(nodes_ff),
                                                  nbasis = min(15, length(nodes_ff)))
                    fd <- Data2fd(
                      argvals = nodes_ff,
                      y = t(data),
                      basisobj = basis
                    )
                    ff <- funFEM::funFEM(fd, K = params$n_clust)
                    list(cluster = ff$cls, iter = NA)
                  },
                  fda_usc = {
                    fd <- fda.usc::fdata(data,
                                         argvals = nodes,
                                         rangeval = c(min(nodes), max(nodes)))
                    cl <- fda.usc::kmeans.fd(
                      fd,
                      ncl = params$n_clust,
                      par.init = list(method = "centers", centers = fd[init_centers])
                    )
                    list(cluster = cl$cluster,
                         iter = cl$iter)
                  },
                  fdacluster = { 
                    fd <- funData(argvals = list(as.numeric(nodes)), X = data)
                    cl <- fdakmeans(fd, n_clusters = params$n_clust, seeds = init_centers)
                    list(cluster = cl$memberships, iter = cl$n_iterations)
                  }
                )
              }, error = function(e) {
                msg <- sprintf(
                  "Iteration %d failed for %s/%s/%s/%s/%s: %s\n",
                  n,
                  space,
                  recon,
                  type,
                  method,
                  dist,
                  e$message
                )
                cat(msg)
                return(NULL)
              })
            }
            if (space == "2d") {
              result <- tryCatch({
                switch(method, mv = {
                  km <- kmeans(data, centers = data[init_centers, ])
                  list(cluster = km$cluster,
                       iter = km$iter)
                }, tensor_pca = {
                  xvals <- unique(sort(nodes[, 1]))
                  yvals <- unique(sort(nodes[, 2]))
                  
                  range_x <- range(xvals)
                  range_y <- range(yvals)
                  
                  nbasis_x <- 10
                  nbasis_y <- 10
                  
                  basis_x <- create.bspline.basis(range_x, nbasis = nbasis_x)
                  basis_y <- create.bspline.basis(range_y, nbasis = nbasis_y)
                  
                  Phi_x <- eval.basis(xvals, basis_x)
                  Phi_y <- eval.basis(yvals, basis_y)
                  
                  Phi <- kronecker(Phi_y, Phi_x)
                  
                  n_obs <- nrow(data)
                  n_basis <- ncol(Phi)
                  coef_mat <- matrix(NA, n_obs, n_basis)
                  
                  for (i in 1:n_obs) {
                    y_i <- data[i, ]
                    coef_mat[i, ] <- lm.fit(Phi, y_i)$coefficients
                  }
                  
                  pca_res <- prcomp(coef_mat, scale. = TRUE)
                  scores <- pca_res$x[, 1:5]
                  
                  km <- kmeans(scores, centers = scores[init_centers, ])
                  
                  list(cluster = km$cluster,
                       iter = km$iter)
                })
              }, error = function(e) {
                msg <- sprintf(
                  "Iteration %d failed for %s/%s/%s/%s/%s: %s\n",
                  n,
                  space,
                  recon,
                  type,
                  method,
                  dist,
                  e$message
                )
                cat(msg)
                return(NULL)
              })
            }
            if (space == "3d") {
              result <- tryCatch({
                switch(method, mv = {
                  km <- kmeans(data, centers = data[init_centers, ])
                  list(cluster = km$cluster,
                       iter = km$iter)
                }, tensor_pca = {
                  xvals <- unique(sort(nodes[, 1]))
                  yvals <- unique(sort(nodes[, 2]))
                  zvals <- unique(sort(nodes[, 3]))
                  
                  range_x <- range(xvals)
                  range_y <- range(yvals)
                  range_z <- range(zvals)
                  
                  nbasis_x <- 10
                  nbasis_y <- 10
                  nbasis_z <- 10
                  
                  basis_x <- create.bspline.basis(range_x, nbasis = nbasis_x)
                  basis_y <- create.bspline.basis(range_y, nbasis = nbasis_y)
                  basis_z <- create.bspline.basis(range_z, nbasis = nbasis_z)
                  
                  Phi_x <- eval.basis(xvals, basis_x)
                  Phi_y <- eval.basis(yvals, basis_y)
                  Phi_z <- eval.basis(zvals, basis_z)
                  
                  Phi <- kronecker(Phi_z, kronecker(Phi_y, Phi_x))
                  
                  n_obs <- nrow(data)
                  n_basis <- ncol(Phi)
                  coef_mat <- matrix(NA, n_obs, n_basis)
                  
                  for (i in 1:n_obs) {
                    y_i <- data[i, ]
                    coef_mat[i, ] <- lm.fit(Phi, y_i)$coefficients
                  }
                  
                  pca_res <- prcomp(coef_mat, scale. = TRUE)
                  scores <- pca_res$x[, 1:5]
                  
                  km <- kmeans(scores, centers = scores[init_centers, ])
                  
                  list(cluster = km$cluster,
                       iter = km$iter)
                })
              }, error = function(e) {
                msg <- sprintf(
                  "Iteration %d failed for %s/%s/%s/%s/%s: %s\n",
                  n,
                  space,
                  recon,
                  type,
                  method,
                  dist,
                  e$message
                )
                cat(msg)
                return(NULL)
              })
            }
            end_time <- Sys.time()
            elapsed <- as.numeric(difftime(end_time, start_time, units = "secs"))
            if (is.null(result))
              next
            # Save results
            write(
              t(result$cluster),
              memb_path,
              sep = ",",
              ncolumns = length(result$cluster),
              append = TRUE
            )
            
            msg <- sprintf(
              "%s/%s/%s/%s/%s: iteration %d completed in %.3f seconds, iter: %s\n",
              space,
              recon,
              type,
              method,
              dist,
              n,
              elapsed,
              ifelse(is.na(result$iter), "NA", result$iter)
            )
            cat(msg)
          }
        }
      }
    }
  }
}
