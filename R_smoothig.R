# use tps to smooth the datasets
library(mgcv)

# --- Parameters ---
output_dir <- "./data"
data_dir <- "./data"

log_file <- "./data/gen_log.txt"
comp_log_file <- "./comp_log.txt"
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
N <- params$N

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

# 1D
for (dims in c("1d", "2d", "3d")) { #
  temp_path <- file.path(output_dir, dims, "smooth")
  dir.create(temp_path, recursive = TRUE, showWarnings = FALSE)
  nodes_path <- file.path(data_dir, dims, "nodes.csv")
  if (!file.exists(nodes_path)) {
    cat(paste0("Nodes file ", nodes_path, " does not exist, skipping...\n"))
    next
  }
  nodes <- as.matrix(read.csv(nodes_path, header = FALSE)) # [n_nodes, dims]
  
  for (type in c("gauss")) { #"wave", , "warped_bump", "spline_like"
    type_path <- file.path(temp_path, type)
    dir.create(type_path, recursive = TRUE, showWarnings = FALSE)
    
    for (i in 0:(N - 1)) {
      input_path <- file.path(data_dir, dims, type, paste0(type, "_", i, ".csv"))
      if (!file.exists(input_path)) {
        cat(paste0(
          "Input file ",
          input_path,
          " does not exist, skipping...\n"
        ))
        next
      }
      output_path <- file.path(type_path, paste0(type, "_", i, ".csv"))
      data <- as.matrix(read.csv(input_path, header = FALSE))   # [n_samples, n_nodes]
      reconstructed <- matrix(NA, nrow = nrow(data), ncol = nrow(nodes))
      if (dims == "1d") {
        for (j in 1:nrow(data)) {
          y <- data[j, ]
          observed <- !is.na(y)
          if (sum(observed) >= 4) {
            df <- data.frame(x = nodes[observed, 1], y_obs = y[observed])
            fit <- gam(y_obs ~ s(x, bs = "tp", k = 10), data = df, select = TRUE)
            pred_df <- data.frame(x = nodes[, 1])
            reconstructed[j, ] <- predict(fit, newdata = pred_df)
          }
        }
      }
      if (dims == "2d") {
        for (j in 1:nrow(data)) {
          y <- data[j, ]
          obs <- !is.na(y)
          if (sum(obs) >= 4) {
            df <- data.frame(x = nodes[obs, 1],
                             y = nodes[obs, 2],
                             z = y[obs])
            model <- gam(z ~ s(x, y, bs = "tp", k = 100), data = df, select = TRUE)
            pred_df <- data.frame(x = nodes[, 1], y = nodes[, 2])
            reconstructed[j, ] <- predict(model, newdata = pred_df)
          }
        }
      }
      if (dims == "3d") {
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
            model <- gam(value ~ s(x, y, z, bs = "tp", k = 300), data = df, select = TRUE)
            pred_df <- data.frame(x = nodes[, 1],
                                  y = nodes[, 2],
                                  z = nodes[, 3])
            reconstructed[j, ] <- predict(model, newdata = pred_df)
          }
        }
      }
      cat(paste0(file.path(dims, type), ": iteration ", i, " completed\n"))
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
