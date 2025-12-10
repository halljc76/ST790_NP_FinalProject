# generate_synthetic_data.R
# Goal: generate synthetic data for image response regression via
# deep neural networks
source("utils.R")
source("design.R")

# set seed
set_seed <- function(seed=NULL){
  if(!is.null(seed)){
    set.seed(seed)
  }
}

# generate synthetic data
generate_synthetic_data <- function(
    n_voxels=128,
    n_indivs=20,
    n_features=3,
    beta_stn=0.10,
    omega_stn=0.05,
    noise_dist='gauss',
    seed=NULL,
    out_file='data.rds'
){
  set_seed(seed)
  
  img_shape <- c(n_voxels, n_voxels, n_voxels)
  
  data <- gen_data(
    V_out=img_shape,
    N=n_indivs,
    Q=n_features,
    beta_stn=beta_stn,
    omega_stn=omega_stn,
    omega_itv=1.0,
    noise_dist=noise_dist,
    noise_var='wave',
    scale=1.0,
    cut=TRUE
  )
  
  save_pickle(data, out_file)
  
  return(data)
}

#------------------------------
# Example usage
#------------------------------
# data <- generate_synthetic_data(
#   n_voxels=64,
#   n_indivs=10,
#   n_features=2,
#   beta_stn=0.1,
#   omega_stn=0.05,
#   seed=123,
#   out_file='synthetic_data.rds'
# )
