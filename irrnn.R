# irrnn.R
# Image Response Regression via Deep Neural Networks (SVCM framework)

library(torch)
source("utils.R")
source("train.R")

# --- SemiHardshrink activation module ---
SemiHardshrink <- nn_module(
  "SemiHardshrink",
  initialize = function(lambd, alpha = 0.1) {
    self$alpha      <- alpha
    self$activation <- nn_hardshrink(lambd)
  },
  forward = function(x) {
    y <- self$activation(x)
    if (!is.null(self$alpha) && self$training) {
      y <- x * self$alpha + y * (1 - self$alpha)
    }
    y
  }
)

# --- Single MLP with multi-output head ---
MLP <- nn_module(
  "MLP",
  initialize = function(widths, activation, lr, shrink = FALSE, l2pen = NULL) {
    activation_dict <- list(
      relu    = nn_relu(inplace = TRUE),
      leaky   = nn_leaky_relu(0.1, inplace = TRUE),
      swish   = nn_silu(inplace = TRUE),
      sigmoid = nn_sigmoid()
    )
    self$activation_func <- activation_dict[[activation]]
    
    layers <- list()
    for (i in seq_len(length(widths) - 1)) {
      layers <- append(layers, nn_linear(widths[i], widths[i + 1]))
      if (i < length(widths) - 1) {
        layers <- append(layers, self$activation_func)
      }
    }
    self$net <- nn_sequential(!!!layers)
    
    self$shrinker <- if (shrink) SemiHardshrink(lambd = 1.0, alpha = 0.1) else NULL
    self$lr       <- lr
    self$l2pen    <- l2pen
  },
  forward = function(x, threshold = NULL) {
    x <- self$net(x)  # shape: [batch, n_out]
    if (!is.null(threshold) && !is.null(self$shrinker)) {
      # threshold should broadcast to same shape as x
      x <- x / threshold
      x <- self$shrinker(x)
      x <- x * threshold
    }
    x
  }
)

# --- Dataset class ---
ImageDataset <- dataset(
  name = "ImageDataset",
  initialize = function(value, coord = NULL, img_shape = NULL, threshold = NULL) {
    # value: matrix [n_voxels, n_out]
    if (is.null(coord)) {
      if (is.null(img_shape)) stop("Either coord or img_shape must be provided.")
      coord <- get_coord(img_shape)
    }
    self$coord <- torch_tensor(coord, dtype = torch_float())
    self$value <- torch_tensor(value, dtype = torch_float())
    self$threshold <- if (!is.null(threshold)) {
      torch_tensor(threshold, dtype = torch_float())
    } else {
      NULL
    }
  },
  .getitem = function(i) {
    if (is.null(self$threshold)) {
      list(self$coord[i, ], self$value[i, ])
    } else {
      list(list(self$coord[i, ], self$threshold[i, ]), self$value[i, ])
    }
  },
  .length = function() {
    self$coord$size(1)
  }
)

# --- Utility functions ---
get_coord <- function(shape) {
  grids <- lapply(shape, function(n) seq(0, 1, length.out = n))
  arr   <- do.call(expand.grid, grids)
  as.matrix(arr)   # [n_voxels, dim]
}

normalize_coordinate <- function(x) {
  x <- x - min(x)
  x / (max(x) + 1e-12)
}

hard_threshold <- function(u, eta, B) {
  # u: matrix or vector of NN outputs
  sign(u) * (abs(u) >= eta) * pmin(abs(u), B)
}

get_ols_est <- function(x, y) {
  xtx     <- t(x) %*% x
  xtx_inv <- solve(xtx)
  xty     <- t(x) %*% y
  est     <- xtx_inv %*% xty
  list(est = est, xtx_inv = xtx_inv)
}

# --- Fit MLP: training via minibatch SGD, prediction in one pass ---
fit <- function(value, hidden_widths, activation, lr, batch_size, epochs,
                coord = NULL, img_shape = NULL, threshold = NULL, l2pen = NULL,
                device = NULL, prefix = "") {
  
  # value: matrix [n_voxels, n_out]
  dataset <- ImageDataset(value, coord = coord, img_shape = img_shape,
                          threshold = threshold)
  
  # input dimension (# of spatial coordinates)
  img_ndim <- if (is.null(coord)) length(img_shape) else ncol(as_array(dataset$coord))
  n_out    <- ncol(as_array(dataset$value))
  
  widths <- c(img_ndim, hidden_widths, n_out)
  
  if (is.null(device)) {
    device <- if (cuda_is_available()) "cuda" else "cpu"
  }
  
  model <- get_model(
    model_class  = MLP,
    model_kwargs = list(
      widths    = widths,
      lr        = lr,
      activation = activation,
      shrink    = !is.null(threshold),
      l2pen     = l2pen
    ),
    dataset     = dataset,
    prefix      = prefix,
    epochs      = epochs,
    load_existing = FALSE,
    device      = device,
    batch_size  = batch_size
  )
  
  model$eval()
  
  # Predict over ALL coordinates in one shot
  coord_array <- as_array(dataset$coord)
  x_all       <- torch_tensor(coord_array, dtype = torch_float(),
                              device = device)
  
  if (!is.null(threshold)) {
    threshold_array <- as_array(dataset$threshold)
    t_all <- torch_tensor(threshold_array, dtype = torch_float(),
                          device = device)
    y_pred <- model(x_all, t_all)
  } else {
    y_pred <- model(x_all)
  }
  
  y_pred <- y_pred$to(device = torch_device("cpu"))
  y_pred <- y_pred$detach()
  as_array(y_pred)  # matrix [n_voxels, n_out]
}

# --- Full SVCM estimation pipeline (one-step, as in the paper) ---
estimate_svc_model <- function(x, y, s = NULL, img_shape = NULL,
                               hidden_widths = rep(256, 4),
                               activation = "leaky",
                               lr = 1e-3, batch_size = 4096,
                               epochs = 50, B = 1e12,
                               eta_beta = NULL, eta_alpha = NULL,
                               device = NULL) {
  
  # x: N x J ; y: N x V
  # img_shape: original 3D image shape (e.g., c(64,64,64))
  
  # Step 1: main effects β(s)
  ols_est  <- get_ols_est(x, y)
  beta_hat <- ols_est$est              # J x V
  beta_pred <- fit(
    value        = t(beta_hat),        # [V, J]
    hidden_widths = hidden_widths,
    activation   = activation,
    lr           = lr,
    batch_size   = batch_size,
    epochs       = epochs,
    coord        = s,
    img_shape    = img_shape,
    device       = device,
    prefix       = "beta_"
  )
  beta_pred <- t(beta_pred)            # back to J x V
  
  # Step 2: individual deviations α_i(s)
  indiveff_obsr <- y - x %*% beta_pred   # N x V
  alpha_pred <- fit(
    value        = t(indiveff_obsr),     # [V, N]
    hidden_widths = hidden_widths,
    activation   = activation,
    lr           = lr,
    batch_size   = batch_size,
    epochs       = epochs,
    coord        = s,
    img_shape    = img_shape,
    device       = device,
    prefix       = "alpha_"
  )
  alpha_pred <- t(alpha_pred)            # N x V
  
  # Step 3: log σ^2(s)
  residuals      <- y - x %*% beta_pred - alpha_pred    # N x V
  log_sigma_obsr <- log(colMeans(residuals^2))          # length V
  log_sigma_pred <- fit(
    value        = matrix(log_sigma_obsr, ncol = 1),    # [V,1]
    hidden_widths = hidden_widths,
    activation   = activation,
    lr           = lr,
    batch_size   = batch_size,
    epochs       = epochs,
    coord        = s,
    img_shape    = img_shape,
    device       = device,
    prefix       = "logsigma_"
  )
  log_sigma_pred <- as.numeric(log_sigma_pred)          # vector length V
  
  # Step 4: hard threshold on NN outputs
  if (!is.null(eta_beta)) {
    beta_pred <- hard_threshold(beta_pred, eta_beta, B)
  }
  if (!is.null(eta_alpha)) {
    alpha_pred <- hard_threshold(alpha_pred, eta_alpha, B)
  }
  log_sigma_pred <- hard_threshold(log_sigma_pred, 0, B)
  
  list(
    beta      = beta_pred,   # J x V
    alpha     = alpha_pred,  # N x V
    log_sigma = log_sigma_pred,
    s         = s,
    img_shape = img_shape
  )
}
