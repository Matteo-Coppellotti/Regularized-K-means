library(mgcv)
library(INLA)
library(sp)
library(sinkr)

output_dir <- "./data_reconstructed"
data_dir <- "./data"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

#methods
methods <- environment()
methods[["1d"]] <- c("tps", "dineof") #"inla","psplines"    
methods[["2d"]] <- c("tps", "dineof") #"inla","psplines"
methods[["3d"]] <- c("tps", "dineof") #"psplines"

#READ PARAMETERS
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

# Helper function to determine if points are inside a polygon
inSide <- function(bnd, x, y) {
  sp::point.in.polygon(x, y, bnd$x, bnd$y) > 0
}

# Make a regular grid inside a polygon
make_soap_grid <- function(bnd, n = c(25, 25)) {
  x_seq <- seq(min(bnd$x), max(bnd$x), length.out = n[1])
  y_seq <- seq(min(bnd$y), max(bnd$y), length.out = n[2])
  grid <- expand.grid(x = x_seq, y = y_seq)
  inside <- inSide(bnd, grid$x, grid$y)
  grid[inside, , drop = FALSE]
}

# --- Extract parameters ---
params <- extract_log_params("./data/gen_log.txt")
N <- params$N

# 1D
for (dims in c("1d")) {
  temp_path <- file.path(output_dir, dims)
  dir.create(temp_path, recursive = TRUE, showWarnings = FALSE)
  nodes_path <- file.path(data_dir, dims, "nodes.csv")
  if (!file.exists(nodes_path)) {
    cat(paste0("Nodes file ", nodes_path, " does not exist, skipping...\n"))
    next
  }
  nodes <- as.matrix(read.csv(nodes_path, header = FALSE)) # [n_nodes, dims]
  for (miss in c("scatter", "area")) {
    miss_path <- file.path(temp_path, miss)
    dir.create(miss_path, recursive = TRUE, showWarnings = FALSE)
    for (type in c("gauss")) { #"wave", , "warped_bump", "spline_like"
      type_path <- file.path(miss_path, type)
      dir.create(type_path,
                 recursive = TRUE,
                 showWarnings = FALSE)
      for (method in methods[[dims]]) {
        method_path <- file.path(type_path, method)
        if (file.exists(method_path)) {
          unlink(method_path,
                 recursive = TRUE,
                 force = TRUE)
        }
        dir.create(method_path,
                   recursive = TRUE,
                   showWarnings = FALSE)
        if (method == "") {
          next
        }
        for (i in 0:(N - 1)) {
          input_path <- file.path(data_dir, dims, type, paste0(type, "_", i, ".csv"))
          # input_path <- file.path(data_dir, dims, "no_noise", type, paste0(type, "_", i, ".csv"))
          if (!file.exists(input_path)) {
            cat(paste0(
              "Input file ",
              input_path,
              " does not exist, skipping...\n"
            ))
            next
          }
          obs_pattern_path = file.path(data_dir,
                                       dims,
                                       "observation_patterns",
                                       miss,
                                       paste0(miss, "_", i, ".csv"))
          if (!file.exists(obs_pattern_path)) {
            cat(
              paste0(
                "Observation pattern file ",
                obs_pattern_path,
                " does not exist, skipping...\n"
              )
            )
            next
          }
          output_path <- file.path(method_path, paste0(method, "_", i, ".csv"))
          data <- as.matrix(read.csv(input_path, header = FALSE))   # [n_samples, n_nodes]
          obs_pattern <- as.matrix(read.csv(obs_pattern_path, header = FALSE)) # [n_samples, n_nodes]
          data[obs_pattern == 0] = NA
          reconstructed <- matrix(NA, nrow = nrow(data), ncol = nrow(nodes))
          if (method == "tps") {
            for (j in 1:nrow(data)) {
              y <- data[j, ]
              observed <- !is.na(y)
              if (sum(observed) >= 4) {
                df <- data.frame(x = nodes[observed, 1], y_obs = y[observed])
                n_unique_x <- length(unique(df$x))
                safe_k <- max(5, min(n_unique_x - 1, 20))
                fit <- gam(y_obs ~ s(x, bs = "tp", k = safe_k), data = df, select = TRUE)
                pred_df <- data.frame(x = nodes[, 1])
                reconstructed[j, ] <- predict(fit, newdata = pred_df)
              }
            }
          }
          if (method == "inla") {
            mesh <- inla.mesh.1d(seq(min(nodes), max(nodes), length.out = 50))
            spde <- inla.spde2.matern(mesh = mesh)
            
            for (j in 1:nrow(data)) {
              y <- data[j, ]
              observed <- !is.na(y)
              
              if (sum(observed) >= 4) {
                A_obs <- inla.spde.make.A(mesh, loc = nodes[observed])
                stack <- inla.stack(
                  data = list(y = y[observed]),
                  A = list(A_obs),
                  effects = list(field = 1:spde$n.spde),
                  tag = "est"
                )
                
                result <- inla(
                  y ~ -1 + f(field, model = spde),
                  data = inla.stack.data(stack),
                  control.predictor = list(A = inla.stack.A(stack), compute = TRUE),
                  control.family = list(hyper = list(
                    prec = list(initial = 1, fixed = FALSE)
                  ))
                )
                
                A_pred <- inla.spde.make.A(mesh, loc = nodes)
                pred <- as.vector(A_pred %*% result$summary.random$field$mean)
                reconstructed[j, ] <- pred
              }
            }
          }
          if (method == "psplines") {
            for (j in 1:nrow(data)) {
              y <- data[j, ]
              observed <- !is.na(y)
              if (sum(observed) >= 4) {
                df <- data.frame(x = nodes[observed, 1], y_obs = y[observed])
                fit <- gam(y_obs ~ s(x, bs = "ps", m = c(2, 2)), data = df)
                newdata <- data.frame(x = nodes[, 1])
                reconstructed[j, ] <- predict(fit, newdata = newdata)
              }
            }
          }
          if (method == "dineof") {
            reconstructed <- dineof(data)$Xa
          }
          cat(paste0(
            file.path(dims, miss, type, method),
            ": iteration ",
            i,
            " completed\n"
          ))
          write.table(
            reconstructed,
            file = output_path,
            row.names = FALSE,
            col.names = FALSE,
            sep = ","
          )
        }
      }
    }
  }
}

#2D
for (dims in c("2d")) {
  temp_path <- file.path(output_dir, dims)
  dir.create(temp_path, recursive = TRUE, showWarnings = FALSE)
  nodes_path <- file.path(data_dir, dims, "nodes.csv")
  if (!file.exists(nodes_path)) {
    cat(paste0("Nodes file ", nodes_path, " does not exist, skipping...\n"))
    next
  }
  nodes <- as.matrix(read.csv(nodes_path, header = FALSE)) # [n_nodes, dims]
  as.numeric(read.csv(nodes_path, header = FALSE)[, 1])
  for (miss in c("scatter", "area")) {
    miss_path <- file.path(temp_path, miss)
    dir.create(miss_path, recursive = TRUE, showWarnings = FALSE)
    for (type in c("gauss")) { #"wave", , "warped_bump"
      type_path <- file.path(miss_path, type)
      dir.create(type_path,
                 recursive = TRUE,
                 showWarnings = FALSE)
      for (method in methods[[dims]]) {
        method_path <- file.path(type_path, method)
        if (file.exists(method_path)) {
          unlink(method_path,
                 recursive = TRUE,
                 force = TRUE)
        }
        dir.create(method_path,
                   recursive = TRUE,
                   showWarnings = FALSE)
        if (method == "") {
          next
        }
        for (i in 0:(N - 1)) {
          input_path <- file.path(data_dir, dims, type, paste0(type, "_", i, ".csv"))
          # input_path <- file.path(data_dir, dims, "no_noise", type, paste0(type, "_", i, ".csv"))
          if (!file.exists(input_path)) {
            cat(paste0(
              "Input file ",
              input_path,
              " does not exist, skipping...\n"
            ))
            next
          }
          obs_pattern_path = file.path(data_dir,
                                       dims,
                                       "observation_patterns",
                                       miss,
                                       paste0(miss, "_", i, ".csv"))
          if (!file.exists(obs_pattern_path)) {
            cat(
              paste0(
                "Observation pattern file ",
                obs_pattern_path,
                " does not exist, skipping...\n"
              )
            )
            next
          }
          output_path <- file.path(method_path, paste0(method, "_", i, ".csv"))
          data <- as.matrix(read.csv(input_path, header = FALSE))   # [n_samples, n_nodes]
          obs_pattern <- as.matrix(read.csv(obs_pattern_path, header = FALSE)) # [n_samples, n_nodes]
          data[obs_pattern == 0] = NA
          reconstructed <- matrix(NA, nrow = nrow(data), ncol = nrow(nodes))
          if (method == "tps") {
            for (j in 1:nrow(data)) {
              y <- data[j, ]
              obs <- !is.na(y)
              if (sum(obs) >= 4) {
                df <- data.frame(x = nodes[obs, 1],
                                 y = nodes[obs, 2],
                                 z = y[obs])
                n_unique_xy <- nrow(unique(df[, c("x", "y")]))
                safe_k <- max(10, min(n_unique_xy - 1, 50))
                model <- gam(z ~ s(x, y, bs = "tp", k = safe_k), data = df, select = TRUE)
                pred_df <- data.frame(x = nodes[, 1], y = nodes[, 2])
                reconstructed[j, ] <- predict(model, newdata = pred_df)
              }
            }
          }
          if (method == "inla") {
            mesh <- inla.mesh.2d(
              loc = nodes,
              max.edge = c(0.1, 0.5),
              cutoff = 0.01
            )
            spde <- inla.spde2.matern(mesh)

            for (j in 1:nrow(data)) {
              y <- data[j, ]
              obs <- !is.na(y)
              if (sum(obs) >= 4) {
                A <- inla.spde.make.A(mesh, loc = nodes[obs, ])
                stack <- inla.stack(
                  data = list(z = y[obs]),
                  A = list(A),
                  effects = list(i = 1:spde$n.spde),
                  tag = "est"
                )
                result <- inla(
                  z ~ -1 + f(i, model = spde),
                  data = inla.stack.data(stack),
                  control.predictor = list(A = inla.stack.A(stack), compute = TRUE)
                )
                A_pred <- inla.spde.make.A(mesh, loc = nodes)
                pred_vals <- as.vector(A_pred %*% result$summary.random$i$mean)
                reconstructed[j, ] <- pred_vals
              }
            }
          }
          if (method == "psplines") {
            for (j in 1:nrow(data)) {
              y <- data[j, ]
              obs <- !is.na(y)
              if (sum(obs) >= 4) {
                df <- data.frame(x = nodes[obs, 1], y = nodes[obs, 2], z = y[obs])
                # P-splines smooth in 2D
                model <- gam(z ~ te(x, y, bs = c("ps","ps"), m = c(2, 2)), data = df)
                pred_df <- data.frame(x = nodes[, 1], y = nodes[, 2])
                reconstructed[j, ] <- predict(model, newdata = pred_df)
              }
            }
          }
          if (method == "dineof") {
            reconstructed <- dineof(data)$Xa
          }
          cat(paste0(
            file.path(dims, miss, type, method),
            ": iteration ",
            i,
            " completed\n"
          ))
          write.table(
            reconstructed,
            file = output_path,
            row.names = FALSE,
            col.names = FALSE,
            sep = ","
          )
        }
      }
    }
  }
}

#3D
for (dims in c("3d")) {
  temp_path <- file.path(output_dir, dims)
  dir.create(temp_path, recursive = TRUE, showWarnings = FALSE)
  nodes_path <- file.path(data_dir, dims, "nodes.csv")
  if (!file.exists(nodes_path)) {
    cat(paste0("Nodes file ", nodes_path, " does not exist, skipping...\n"))
    next
  }
  nodes <- as.matrix(read.csv(nodes_path, header = FALSE)) # [n_nodes, dims]
  as.numeric(read.csv(nodes_path, header = FALSE)[, 1])
  for (miss in c("scatter", "area")) {
    miss_path <- file.path(temp_path, miss)
    dir.create(miss_path, recursive = TRUE, showWarnings = FALSE)
    for (type in c("gauss")) { #"wave", 
      type_path <- file.path(miss_path, type)
      dir.create(type_path,
                 recursive = TRUE,
                 showWarnings = FALSE)
      for (method in methods[[dims]]) {
        method_path <- file.path(type_path, method)
        if (file.exists(method_path)) {
          unlink(method_path,
                 recursive = TRUE,
                 force = TRUE)
        }
        dir.create(method_path,
                   recursive = TRUE,
                   showWarnings = FALSE)
        if (method == "") {
          next
        }
        for (i in 0:(N - 1)) {
          input_path <- file.path(data_dir, dims, type, paste0(type, "_", i, ".csv"))
          # input_path <- file.path(data_dir, dims, "no_noise", type, paste0(type, "_", i, ".csv"))
          if (!file.exists(input_path)) {
            cat(paste0(
              "Input file ",
              input_path,
              " does not exist, skipping...\n"
            ))
            next
          }
          obs_pattern_path = file.path(data_dir,
                                       dims,
                                       "observation_patterns",
                                       miss,
                                       paste0(miss, "_", i, ".csv"))
          if (!file.exists(obs_pattern_path)) {
            cat(
              paste0(
                "Observation pattern file ",
                obs_pattern_path,
                " does not exist, skipping...\n"
              )
            )
            next
          }
          output_path <- file.path(method_path, paste0(method, "_", i, ".csv"))
          data <- as.matrix(read.csv(input_path, header = FALSE))   # [n_samples, n_nodes]
          obs_pattern <- as.matrix(read.csv(obs_pattern_path, header = FALSE)) # [n_samples, n_nodes]
          data[obs_pattern == 0] = NA
          reconstructed <- matrix(NA, nrow = nrow(data), ncol = nrow(nodes))
          if (method == "tps") {
            for (j in 1:nrow(data)) {
              y <- data[j, ]
              obs <- !is.na(y)
              if (sum(obs) >= 4) {
                df <- data.frame(
                  x = nodes[obs, 1],
                  y = nodes[obs, 2],
                  z = nodes[obs, 3],
                  value = y[obs]
                )
                n_unique_xyz <- nrow(unique(df[, c("x", "y", "z")]))
                safe_k <- max(20, min(n_unique_xyz - 1, 100))
                model <- gam(value ~ s(x, y, z, bs = "tp", k = safe_k), data = df, select = TRUE)
                pred_df <- data.frame(x = nodes[, 1],
                                      y = nodes[, 2],
                                      z = nodes[, 3])
                reconstructed[j, ] <- predict(model, newdata = pred_df)
              }
            }
          }
          if (method == "inla") {
            mesh <- inla.mesh.3d(
              loc = nodes,
              cutoff = 0.05,
              max.edge = c(0.1, 0.5)
            )
            spde <- inla.spde2.matern(mesh)

            for (j in 1:nrow(data)) {
              y <- data[j, ]
              obs <- !is.na(y)
              if (sum(obs) >= 4) {
                A <- inla.spde.make.A(mesh = mesh, loc = nodes[obs, ])
                stack <- inla.stack(
                  data = list(z = y[obs]),
                  A = list(A),
                  effects = list(i = 1:spde$n.spde),
                  tag = "est"
                )
                result <- inla(
                  z ~ -1 + f(i, model = spde),
                  data = inla.stack.data(stack),
                  control.predictor = list(A = inla.stack.A(stack), compute = TRUE)
                )
                A_pred <- inla.spde.make.A(mesh = mesh, loc = nodes)
                pred_vals <- as.vector(A_pred %*% result$summary.random$i$mean)
                reconstructed[j, ] <- pred_vals
              }
            }
          }
          if (method == "psplines") {
            for (j in 1:nrow(data)) {
              y <- data[j, ]
              obs <- !is.na(y)
              if (sum(obs) >= 4) {
                df <- data.frame(
                  x = nodes[obs, 1],
                  y = nodes[obs, 2],
                  z = nodes[obs, 3],
                  value = y[obs]
                )
                model <- gam(value ~ te(x, y, z, bs = c("ps","ps","ps"), m = c(2, 2)), data = df)
                pred_df <- data.frame(x = nodes[, 1], y = nodes[, 2], z = nodes[, 3])
                reconstructed[j, ] <- predict(model, newdata = pred_df)
              }
            }
          }
          if (method == "dineof") {
            reconstructed <- dineof(data)$Xa
          }
          cat(paste0(
            file.path(dims, miss, type, method),
            ": iteration ",
            i,
            " completed\n"
          ))
          write.table(
            reconstructed,
            file = output_path,
            row.names = FALSE,
            col.names = FALSE,
            sep = ","
          )
        }
      }
    }
  }
}
