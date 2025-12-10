# run_one_replicate.R
source("generate_synthetic_data.R")
source("run_irrnn.R")
source("metrics.R")

run_one_replicate <- function(beta_stn, n_indivs, n_voxels,
                              rep_id = 1, outdir = ".", seed = 2025) {
  
  if (!is.null(seed)) set.seed(seed + rep_id)
  
  # generate synthetic dataset
  datafile <- file.path(outdir, sprintf("data_rep%02d.rds", rep_id))
  
  data <- generate_synthetic_data(
    n_voxels = n_voxels,
    n_indivs = n_indivs,
    n_features = 3, 
    beta_stn = beta_stn,
    omega_stn = 0.05,
    seed = seed + rep_id,
    out_file = datafile
  )
  
  # fit irrnn
  prefix <- file.path(outdir, sprintf("rep%02d_", rep_id))
  
  pred <- run_irrnn_synthetic(
    data_file = datafile,
    prefix     = prefix,
    hidden_widths = rep(64, 2),
    activation = "leaky",
    lr = 1e-3,
    batch_size = 1024,
    epochs     = 10,
    seed = seed + rep_id
  )
  
  # compute eval metrics
  truth <- readRDS(paste0(prefix, "true.rds"))
  
  metrics <- list(
    beta_mse  = mean_squared_difference(truth$maineff, pred$beta),
    alpha_mse = mean_squared_difference(truth$indiveff, pred$alpha),
    sigma_mse = mean_squared_difference(truth$noiselogvar, pred$log_sigma),
    
    beta_cor  = correlation_pearson(c(truth$maineff), c(pred$beta)),
    alpha_cor = correlation_pearson(c(truth$indiveff), c(pred$alpha)),
    sigma_cor = correlation_pearson(c(truth$noiselogvar), c(pred$log_sigma))
  )
  
  outfile <- file.path(outdir, sprintf("metrics_rep%02d.rds", rep_id))
  saveRDS(metrics, outfile)
  
  return(metrics)
}
