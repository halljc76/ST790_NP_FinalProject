library(ggplot2)
library(reshape2)

# Convert a vector back to a 3D array
vec_to_array <- function(v, img_shape) {
  array(v, dim = img_shape)
}

# Extract mid-slices for visualization
get_mid_slices <- function(arr3d) {
  nx = dim(arr3d)[1]
  ny = dim(arr3d)[2]
  nz = dim(arr3d)[3]
  
  list(
    sagittal  = arr3d[floor(nx/2), , ],
    coronal   = arr3d[ , floor(ny/2), ],
    axial     = arr3d[ , , floor(nz/2)]
  )
}

# Plot a single 2D slice
plot_slice <- function(slice2d, title = "") {
  df <- melt(slice2d)
  colnames(df) <- c("x", "y", "value")
  
  ggplot(df, aes(x = x, y = y, fill = value)) +
    geom_raster() +
    scale_fill_gradient2(low="blue", high="red", mid="white") +
    coord_fixed() +
    ggtitle(title) +
    theme_minimal()
}

# Master function: plot β, α, σ² for ONE replicate
plot_replicate_results <- function(rep_dir, rep_id = 1) {
  pred <- readRDS(file.path(rep_dir, sprintf("rep%02d_", rep_id), "pred.rds"))
  truth <- readRDS(file.path(rep_dir, sprintf("rep%02d_", rep_id), "true.rds"))
  
  img_shape <- truth$img_shape
  
  # 1) Choose covariate j = 1
  beta_pred_j  <- vec_to_array(pred$beta[1, ], img_shape)
  beta_true_j  <- vec_to_array(truth$maineff[1, ], img_shape)
  
  # 2) Choose subject i = 1
  alpha_pred_i <- vec_to_array(pred$alpha[1, ], img_shape)
  alpha_true_i <- vec_to_array(truth$indiveff[1, ], img_shape)
  
  # 3) Noise variance
  sigma_pred   <- vec_to_array(pred$log_sigma, img_shape)
  sigma_true   <- vec_to_array(truth$noiselogvar, img_shape)
  
  # Prepare plots
  slices <- list(
    pred_beta  = get_mid_slices(beta_pred_j),
    true_beta  = get_mid_slices(beta_true_j),
    pred_alpha = get_mid_slices(alpha_pred_i),
    true_alpha = get_mid_slices(alpha_true_i),
    pred_sigma = get_mid_slices(sigma_pred),
    true_sigma = get_mid_slices(sigma_true)
  )
  
  # Return a list of ggplot objects
  plots <- list()
  for (name in names(slices)) {
    for (slice_name in names(slices[[name]])) {
      title <- paste(name, slice_name, sep=": ")
      plots[[title]] <- plot_slice(slices[[name]][[slice_name]], title)
    }
  }
  return(plots)
}
