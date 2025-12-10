# run_irrnn.R
library(torch)
source("utils.R")
source("irrnn.R")

set_seed <- function(seed = NULL) {
  if (!is.null(seed)) {
    set.seed(seed)
    torch_manual_seed(seed)
  }
}

run_irrnn_synthetic <- function(
    data_file,
    prefix       = "irrnn_run/",
    hidden_widths = rep(256, 4),
    activation   = "leaky",
    lr           = 1e-3,
    batch_size   = 4096,
    epochs       = 50,
    B            = 1e12,
    eta_beta     = NULL,
    eta_alpha    = NULL,
    seed         = NULL,
    device       = NULL
) {
  dir.create(prefix, showWarnings = FALSE, recursive = TRUE)
  set_seed(seed)
  
  if (is.null(device)) {
    device <- if (cuda_is_available()) "cuda" else "cpu"
  }
  
  data <- load_pickle(data_file)
  y    <- data[[1]]   # N x Vx x Vy x Vz
  x    <- data[[2]]   # N x J
  
  img_shape <- dim(y)[-1]
  y_mat     <- matrix(y, nrow = dim(y)[1], ncol = prod(dim(y)[-1]))
  s         <- NULL   # using regular grid coordinates from img_shape
  
  cat("Fitting SVCM model via deep neural networks...\n")
  t0 <- Sys.time()
  pred <- estimate_svc_model(
    x          = x,
    y          = y_mat,
    s          = s,
    img_shape  = img_shape,
    hidden_widths = hidden_widths,
    activation = activation,
    lr         = lr,
    batch_size = batch_size,
    epochs     = epochs,
    B          = B,
    eta_beta   = eta_beta,
    eta_alpha  = eta_alpha,
    device     = device
  )
  cat("Time elapsed:", as.numeric(Sys.time() - t0), "seconds\n")
  
  save_pickle(pred, paste0(prefix, "pred.rds"))
  
  truth <- list(
    x         = x,
    y         = y_mat,
    s         = s,
    img_shape = img_shape
  )
  if (length(data) >= 5) {
    truth$maineff    <- matrix(data[[3]], nrow = dim(data[[3]])[1])
    truth$indiveff   <- matrix(data[[4]], nrow = dim(data[[4]])[1])
    truth$noiselogvar <- log(matrix(data[[5]], nrow = dim(data[[5]])[1]))
  }
  save_pickle(truth, paste0(prefix, "true.rds"))
  
  cat("Done! Predictions and truth saved to", prefix, "\n")
  pred
}
