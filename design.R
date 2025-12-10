# design.R
# goal: set up the space for simulated data
source("utils.R")  # load helper functions from utils.R
library(abind)

# random cube setup
random_cube <- function(v0, v1, v2, lohi=c(1,2), mima=c(0.2,0.6)) {
  size <- c(v0, v1, v2)
  r <- runif(1, mima[1], mima[2])
  start <- runif(2) * (1 - r)
  r_len <- round(r * size[1:2])
  start_idx <- floor(start * size[1:2])
  end_idx <- start_idx + r_len
  a <- runif(1, lohi[1], lohi[2]) * sample(c(-1,1),1)
  
  x <- array(0, dim=size)
  
  idx_i <- seq(0,1,length.out=(end_idx[1]-start_idx[1]))
  idx_j <- seq(0,1,length.out=(end_idx[2]-start_idx[2]))
  
  for(ii in seq_along(idx_i)) {
    for(jj in seq_along(idx_j)) {
      t_val <- sqrt((idx_i[ii]-0.5)^2 + (idx_j[jj]-0.5)^2)
      t_val <- min(t_val, quantile(c(idx_i, idx_j),0.6))
      t_val <- (t_val - 0)/(max(t_val)+1e-12)
      t_val <- t_val * a * 0.5 + a * 0.5
      x[start_idx[1]+ii, start_idx[2]+jj, ] <- rep(t_val, size[3])
    }
  }
  return(x)
}


# random ellipsoid
random_ellipsoid <- function(v0, v1, v2, lohi=c(2,4), mima=c(0.3,0.7), seed=NULL) {
  if(!is.null(seed)) set.seed(seed)
  size <- c(v0, v1, v2)
  x <- array(0, dim=size)
  r <- runif(1, mima[1], mima[2])
  start <- runif(2) * (1-r)
  end <- start + r
  c <- (start+end)/2
  r <- end - c
  start_idx <- floor(start*size[1:2])
  end_idx <- ceiling(end*size[1:2])
  a <- runif(1, lohi[1], lohi[2]) * sample(c(-1,1),1)
  
  idx_i <- seq(start_idx[1]+1, end_idx[1])
  idx_j <- seq(start_idx[2]+1, end_idx[2])
  
  for(ii in seq_along(idx_i)){
    for(jj in seq_along(idx_j)){
      s_val <- sqrt(sum(((c((start_idx[1]+ii-1)/size[1],
                            (start_idx[2]+jj-1)/size[2]) - c)/r)^2))
      s_val <- (1-s_val)*a
      x[start_idx[1]+ii, start_idx[2]+jj, ] <- rep(s_val, size[3])
    }
  }
  return(x)
}

# random cell
random_cell <- function(size) {
  stopifnot(size[1] %% 2 == 0, size[2] %% 2 == 0)
  size2 <- c(size[1]/2, size[2]/2, size[3])
  x_list <- array(0, dim=c(2,2,size2))
  sign_list <- matrix(c(-1,1,1,-1), 2,2)
  sign_list <- sign_list * sample(c(-1,1),1)
  
  # Fill x_list[0,0] and [0,1] with cubes
  for(i in 1:2){
    t <- random_cube(size2[1], size2[2], size2[3], 
                     lohi=c(1,2), mima=c(0.4,0.4))
    t <- abs(t)/max(abs(t))
    x_list[1,1,, ,] <- pmax(x_list[1,1,, ,], t)
  }
  
  for(i in 1:8){
    t <- random_cube(size2[1], size2[2], size2[3], 
                     lohi=c(1,2), mima=c(0.1,0.1))
    t <- abs(t)/max(abs(t))
    x_list[1,2,, ,] <- pmax(x_list[1,2,, ,], t)
  }
  
  # Fill x_list[1,1] and [1,2] with ellipsoids
  for(i in 1:2){
    t <- random_ellipsoid(size2[1], size2[2], size2[3], 
                          lohi=c(7,8), mima=c(0.5,0.5))
    t <- abs(t)/max(abs(t))
    x_list[2,1,, ,] <- pmax(x_list[2,1,, ,], t)
  }
  
  for(i in 1:8){
    t <- random_ellipsoid(size2[1], size2[2], size2[3], 
                          lohi=c(7,8), mima=c(0.2,0.2))
    t <- abs(t)/max(abs(t))
    x_list[2,2,, ,] <- pmax(x_list[2,2,, ,], t)
  }
  
  # Apply sign and combine quadrants
  x_list <- x_list / sd(x_list)
  sign_array <- array(sign_list, dim = dim(x_list))
  x_list <- x_list * sign_array
  x <- rbind(cbind(x_list[1,1,, ,], x_list[1,2,, ,]),
             cbind(x_list[2,1,, ,], x_list[2,2,, ,]))
  return(x)
}


# paraboloid
random_paraboloid <- function(size, cut=FALSE, lohi=NULL, seed=NULL) {
  if(!is.null(seed)) set.seed(seed)
  c_vec <- runif(3)
  a <- 1.0; b <- 1.0
  u <- if(is.null(lohi)) rnorm(1)*a else runif(1, lohi[1], lohi[2])
  
  grid <- expand.grid(x=0:(size[1]-1), y=0:(size[2]-1), z=0:(size[3]-1))
  t <- sweep(as.matrix(grid)/matrix(rep(size, each=nrow(grid)), 
                                    ncol=3), 2, c_vec)
  x <- sqrt(rowSums(t[,1:2]^2))*2*b
  x <- x - mean(x) + u
  x_arr <- array(x, dim=size)
  
  if(cut){
    lower <- floor(runif(3)*0.8*size)
    upper <- lower + floor(0.2*size)
    x_arr[1:lower[1], 1:lower[2], ] <- 0
    x_arr[1:lower[1], (upper[2]+1):size[2], ] <- 0
    x_arr[(upper[1]+1):size[1], 1:lower[2], ] <- 0
    x_arr[(upper[1]+1):size[1], (upper[2]+1):size[2], ] <- 0
  }
  x_arr[!is.na(x_arr)] <- x_arr[!is.na(x_arr)] - mean(x_arr[!is.na(x_arr)])
  return(x_arr)
}

#adjust variance
adj_variance <- function(A, beta, omega, sigmasq, beta_stn, omega_stn, omega_itv, scale=1.0) {
  beta <- beta / sd(A %*% beta) * scale
  omega <- sweep(omega, 1, apply(omega, 1, sd), "/")
  omega <- omega / sd(omega) / sqrt(beta_stn) * sqrt(omega_stn) * scale
  sigmasq <- sigmasq / mean(sigmasq) / beta_stn * scale^2
  return(list(beta=beta, omega=omega, sigmasq=sigmasq))
}

# create wave
gen_wave <- function(shape, frq=2, x0=0, y0=0, cut=FALSE) {
  ndim <- length(shape)
  if(length(frq) == 1) frq <- rep(frq, ndim)
  if(length(x0) == 1) x0 <- rep(x0, ndim)
  if(length(y0) == 1) y0 <- rep(y0, ndim)
  
  out <- array(0, dim=shape)
  
  for(i in 1:(ndim-1)){
    c <- runif(1)
    x <- seq(0, 1, length.out = shape[i])
    y <- sin(2*pi*frq[i]*(x - x0[i] - c)) + y0[i]
    
    # Reshape y to match out's dimensions for broadcasting
    dim_y <- rep(1, ndim)
    dim_y[i] <- length(y)
    y_arr <- array(y, dim=dim_y)
    
    # Repeat along other dimensions to match out
    reps <- shape
    reps[i] <- 1
    y_arr <- array(rep(y_arr, times = prod(reps)), dim = shape)
    
    # Safe addition
    out <- out + y_arr
  }
  
  if(cut){
    is_pos <- out > max(out) * 0.3
    is_neg <- out < min(out) * 0.7
    out[!(is_pos | is_neg)] <- 0
  }
  
  out <- out - min(out)
  out <- out / (max(out) + 1e-12)
  out <- out * 2.0
  return(out)
}

# generate data
gen_data <- function(V_out, N, Q, beta_stn=1.0, omega_stn=1.0,
                     omega_itv=1.0, noise_dist='gauss', noise_var='cons',
                     scale=1.0, Va=128, cut=FALSE, dtype='numeric'){
  
  stopifnot(length(V_out)==3, max(V_out)<=Va)
  A <- matrix(rnorm(N*Q), N, Q)
  V <- c(Va, Va, Va)
  VV <- prod(V)
  beta <- matrix(0, Q, VV)
  for(i in 1:Q) beta[i,] <- as.vector(random_cell(V))
  omega <- matrix(0, N, VV)
  for(i in 1:N) omega[i,] <- as.vector(random_paraboloid(V, cut=cut))
  
  # noise
  if(noise_dist=='gauss'){
    noise <- matrix(rnorm(N*VV), N, VV)
  } else if(noise_dist=='chisq'){
    df <- 3
    noise <- matrix(rchisq(N*VV, df), N, VV)
    noise <- (noise - df)/sqrt(df*2)
  } else stop("noise distribution not recognized")
  noise <- sweep(noise, 2, colMeans(noise))
  noise <- sweep(noise, 2, apply(noise,2,sd), '/')
  
  if(noise_var=='cons'){
    sigmasq <- rep(1, VV)
  } else if(noise_var=='wave'){
    sigmasq <- as.vector(gen_wave(V, cut=cut)^2)
  } else stop("noise variance pattern not recognized")
  
  adj <- adj_variance(A, beta, omega, sigmasq, beta_stn, omega_stn, omega_itv, scale)
  beta <- adj$beta
  omega <- adj$omega
  sigmasq <- adj$sigmasq
  
  X <- A %*% beta + omega + sqrt(sigmasq)*noise
  
  # scaling checks (optional)
  
  stride <- Va %/% V_out
  stop <- stride*V_out
  start <- rep(0,3)
  
  # Ensure arrays are 4D for thin_3d
  X <- array(X, dim=c(N, Va, Va, Va))
  beta <- array(beta, dim=c(Q, Va, Va, Va))
  omega <- array(omega, dim=c(N, Va, Va, Va))
  sigmasq <- array(sigmasq, dim=c(1, Va, Va, Va)) # batch=1 if no samples
  
  X_out <- thin_3d(X, start, stop, stride)
  beta_out <- thin_3d(beta, start, stop, stride)
  omega_out <- thin_3d(omega, start, stop, stride)
  sigmasq_out <- thin_3d(sigmasq, start, stop, stride)
  
  return(list(
    X_out=X_out,
    A=A,
    beta_out=beta_out,
    omega_out=omega_out,
    sigmasq_out=sigmasq_out
  ))
}
