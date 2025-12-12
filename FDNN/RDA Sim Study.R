# Implementing FDNN and competing methods using Fashion-MNIST dataset

# Load the Fashion-MNIST dataset
fashion.mnist_train <- read.csv("~/Downloads/archive-5/fashion-mnist_train.csv")
fashion.mnist_trsamp <- subset(fashion.mnist_train, label == 0 | label == 1 | label == 2)
fashion.mnist_test <- read.csv("~/Downloads/archive-5/fashion-mnist_test.csv")
fashion.mnist_tesamp <- subset(fashion.mnist_test, label == 0 | label == 1 | label == 2)

# Try FDA Multinomial Regression
y = as.integer(fashion.mnist_trsamp$label) 
X = as.matrix(fashion.mnist_trsamp[,-1])
argvals = 1:784

## First perform FPCA
library(refund)
fpca_tr = fpca.face(X)
PC = fpca_tr$scores
df = as.data.frame(cbind(y, PC))
colnames(df) = c('y', paste0("V", 1:6))

fit = gam(list(y ~ s(V1) + s(V2) + s(V3) + s(V4) + s(V5) + s(V6),
               ~ s(V1) + s(V2) + s(V3) + s(V4) + s(V5) + s(V6)),
          family = multinom(K = 2), data = df)
fpca_te = fpca.face(as.matrix(fashion.mnist_tesamp[,-1]))
PC.test = fpca_te$scores
colnames(PC.test) = paste0("V", 1:6)
preds = predict(fit, newdata = as.data.frame(PC.test), type = 'response')
preds_class = apply(preds, 1, which.max) - 1

table(preds_class, fashion.mnist_tesamp$label)
(58 + 187 + 37 + 43 + 165 + 263) / 30

# Next try doing CNN for classification

library(keras3)
install_keras()

# Prepare data
y_train <- to_categorical(y, num_classes = 3)

# If X is flattened images (n x 784), reshape to (28,28,1)
H <- 28; W <- 28
x <- scale(X)
x_train <- array(x, dim = c(nrow(X), H, W, 1))  # (n, H, W, channels)
x_test = array(scale(X.test), dim = c(nrow(X.test), 28, 28, 1))

model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = "relu",
                input_shape = c(H, W, 1)) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 3, activation = "softmax")

model %>% compile(optimizer = "adam",
                  loss = "categorical_crossentropy",
                  metrics = "accuracy")

history <- model %>% fit(x_train, y_train, batch_size = 128, epochs = 15, validation_split = 0.2)

pred <- model %>% predict(x_test)
pred
pred_cnn = apply(pred, 1, which.max) - 1
pred_cnn
table(fashion.mnist_tesamp$label, pred_cnn)
(7 + 25 + 3 + 2 + 28 + 1) / 30

# Lastly apply the FDNN method

library(tensorflow)
library(fda)
train.X = as.matrix(fashion.mnist_trsamp[,-1])
train.y = fashion.mnist_trsamp$label
test.X = as.matrix(fashion.mnist_tesamp[,-1])
test.y = fashion.mnist_tesamp$label

install.packages(c("keras", "tensorflow", "fda", "abind", "caret"))
library(keras3)
library(tensorflow)
library(fda)
library(abind)
library(caret)

# Set seed for reproducibility
set.seed(757)
tf$random$set_seed(757)

# 1. Load and preprocess data
## Normalize pixel values to [0, 1]
train.X <- train.X / 255
test.X <- test.X / 255
## Reshape to (samples, height, width, 1) for functional treatment
x_train <- array_reshape(train.X, c(dim(train.X)[1], 28, 28, 1))
x_test <- array_reshape(test.X, c(dim(test.X)[1], 28, 28, 1))
# One-hot encode labels (10 classes)
y_train <- to_categorical(train.y, 3)
y_test <- to_categorical(test.y, 3)

# 2. Basis expansion (FDR)
# Create B-spline basis functions for 2D functional data
create_basis_system <- function(n_basis = 15) {
  # Define the grid for 28x28 images
  grid <- seq(0, 1, length.out = 28)
  
  # Create B-spline basis for each dimension
  basis <- create.bspline.basis(rangeval = c(0, 1), nbasis = n_basis, norder = 4)
  
  # Evaluate basis functions on the grid
  basis_mat <- eval.basis(grid, basis)
  
  return(basis_mat)
}

# Project images onto basis functions
project_to_basis <- function(images, basis_mat) {
  # images: array of shape (n_samples, 28, 28, 1)
  # basis_mat: basis matrix of shape (28, n_basis)
  
  n_samples <- dim(images)[1]
  n_basis <- dim(basis_mat)[2]
  
  # Remove channel dimension for projection
  images_mat <- array(images, dim = c(n_samples, 28, 28))
  
  # Project along height dimension: X * Phi
  projected <- array(0, dim = c(n_samples, n_basis, 28))
  for (i in 1:n_samples) {
    projected[i, , ] <- t(basis_mat) %*% images_mat[i, , ]
  }
  
  # Project along width dimension: (X * Phi) * Phi^T
  final_projected <- array(0, dim = c(n_samples, n_basis, n_basis))
  for (i in 1:n_samples) {
    final_projected[i, , ] <- projected[i, , ] %*% basis_mat
  }
  
  # Flatten to (n_samples, n_basis * n_basis)
  flattened <- array(final_projected, dim = c(n_samples, n_basis * n_basis))
  
  return(flattened)
}

# Create basis system and project data
n_basis <- 12
basis_mat <- create_basis_system(n_basis)

x_train_proj <- project_to_basis(x_train, basis_mat)
x_test_proj <- project_to_basis(x_test, basis_mat)

# 3. FPCA
perform_fpca <- function(data, variance_explained = 0.95) {
  # Center the data
  data_centered <- scale(data, center = TRUE, scale = FALSE)
  
  # Perform PCA
  pca_result <- prcomp(data_centered, center = FALSE)
  
  # Determine number of components to retain
  variance_prop <- cumsum(pca_result$sdev^2) / sum(pca_result$sdev^2)
  n_components <- which(variance_prop >= variance_explained)[1]
  
  cat(sprintf("Retaining %d components to explain %.1f%% variance\n",
              n_components, variance_explained * 100))
  
  # Extract PC scores
  scores <- pca_result$x[, 1:n_components]
  
  return(list(scores = scores,
              n_components = n_components,
              pca_result = pca_result))
}
fpca_result <- perform_fpca(x_train_proj, variance_explained = 0.95)
x_train_fpca <- fpca_result$scores
x_test_centered <- scale(x_test_proj, 
                         center = attr(scale(x_train_proj, center = TRUE), "scaled:center"),
                         scale = FALSE)
x_test_fpca <- x_test_centered %*% fpca_result$pca_result$rotation[, 1:fpca_result$n_components]

# 4. Build and train DNN
build_dnn_model <- function(input_dim, output_dim = 3) {
  model <- keras_model_sequential() %>%
    # Input layer
    layer_dense(units = 128, activation = 'relu', input_shape = input_dim,
                kernel_regularizer = regularizer_l2(0.001)) %>%
    layer_dropout(rate = 0.5) %>%
    
    # Hidden layers
    layer_dense(units = 64, activation = 'relu',
                kernel_regularizer = regularizer_l2(0.001)) %>%
    layer_dropout(rate = 0.5) %>%
    
    layer_dense(units = 32, activation = 'relu',
                kernel_regularizer = regularizer_l2(0.001)) %>%
    layer_dropout(rate = 0.3) %>%
    
    # Output layer
    layer_dense(units = output_dim, activation = 'softmax')
  
  # Compile model
  model %>% compile(
    optimizer = optimizer_adam(learning_rate = 0.001),
    loss = 'categorical_crossentropy',
    metrics = c('accuracy')
  )
  
  return(model)
}

# Build model
input_dim <- dim(x_train_fpca)[2]
model <- build_dnn_model(input_dim)
summary(model)

# Callbacks
callbacks <- list(
  callback_early_stopping(monitor = 'val_loss', patience = 10, restore_best_weights = TRUE),
  callback_reduce_lr_on_plateau(monitor = 'val_loss', factor = 0.5, patience = 5, min_lr = 1e-6)
)

# Train model
history <- model %>% fit(
  x_train_fpca, y_train,
  epochs = 50,
  batch_size = 32,
  validation_split = 0.1,
  callbacks = callbacks,
  verbose = 1
)

# Evaluate the model
test_eval <- model %>% predict(x_test_fpca)
class_eval = apply(test_eval, 1, which.max)
table(class_eval - 1, fashion.mnist_tesamp$label)
(17 + 38 + 15 + 2 + 32 + 6) / 30

# Overall: CNN does slightly better than FDNN but both perform with < 5% misaccuracy!
