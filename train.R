# train.R
# Goal: train neural network models using torch in R

library(torch)
library(ggplot2)

source("utils.R")  # for save_pickle, load_pickle

MetricTracker <- R6::R6Class(
  "MetricTracker",
  public = list(
    collection = NULL,
    initialize = function() {
      self$collection <- list()
    },
    on_train_epoch_end = function(metrics) {
      self$collection <- c(self$collection, list(metrics))
    },
    clean = function() {
      if (length(self$collection) == 0) return()
      all_keys <- unique(unlist(lapply(self$collection, names)))
      for (i in seq_along(self$collection)) {
        elem <- self$collection[[i]]
        for (key in all_keys) {
          if (!key %in% names(elem)) {
            elem[[key]] <- NA_real_
          } else if (inherits(elem[[key]], "torch_tensor")) {
            elem[[key]] <- as.numeric(elem[[key]]$item())
          }
        }
        self$collection[[i]] <- elem
      }
    }
  )
)

get_model <- function(model_class, model_kwargs, dataset, prefix,
                      epochs = NULL, load_existing = TRUE,
                      device = NULL, batch_size = 128) {
  
  checkpoint_file <- paste0(prefix, "model.pt")
  history_file    <- paste0(prefix, "history.rds")
  
  # Load model if exists (for now: always re-train in this workflow)
  if (load_existing && file.exists(checkpoint_file)) {
    model <- do.call(model_class, model_kwargs)
    model$load_state_dict(torch_load(checkpoint_file))
    cat("Model loaded from", checkpoint_file, "\n")
    history <- load_pickle(history_file)
  } else {
    model   <- NULL
    history <- list()
  }
  
  # Train model
  if (!is.null(epochs) && epochs > 0) {
    train_res <- train_model(
      dataset      = dataset,
      batch_size   = batch_size,
      epochs       = epochs,
      model        = model,
      model_class  = model_class,
      model_kwargs = model_kwargs,
      device       = device
    )
    model   <- train_res$model
    history <- c(history, train_res$history)
    save_pickle(history, history_file)
    cat("History saved to", history_file, "\n")
  }
  
  plot_history(history, prefix)
  return(model)
}

train_model <- function(dataset, batch_size, epochs,
                        model = NULL, model_class = NULL,
                        model_kwargs = list(), device = NULL) {
  
  # Initialize model if not provided
  if (is.null(model)) {
    model <- do.call(model_class, model_kwargs)
  }
  
  if (is.null(device)) {
    device <- if (cuda_is_available()) "cuda" else "cpu"
  }
  model$to(device = device)
  model$train()
  
  dataloader <- dataloader(dataset, batch_size = batch_size, shuffle = TRUE)
  optimizer  <- optim_adam(model$parameters, lr = model_kwargs$lr)
  
  tracker <- MetricTracker$new()
  t0      <- Sys.time()
  
  for (epoch in 1:epochs) {
    total_loss <- 0
    n_batches  <- 0
    
    coro::loop(for (batch in dataloader) {
      optimizer$zero_grad()
      
      # batch: list(input, target) or list(list(input, threshold), target)
      if (is.list(batch[[1]]) && length(batch[[1]]) == 2) {
        input     <- batch[[1]][[1]]$to(device = device)
        threshold <- batch[[1]][[2]]$to(device = device)
        target    <- batch[[2]]$to(device = device)
        output    <- model(input, threshold)
      } else {
        input  <- batch[[1]]$to(device = device)
        target <- batch[[2]]$to(device = device)
        output <- model(input)
      }
      
      loss <- nnf_mse_loss(output, target)
      loss$backward()
      optimizer$step()
      
      total_loss <- total_loss + loss$item()
      n_batches  <- n_batches + 1
    })
    
    avg_loss <- total_loss / n_batches
    tracker$on_train_epoch_end(list(loss = avg_loss))
    cat(sprintf("Epoch %d/%d, loss = %.6f\n", epoch, epochs, avg_loss))
  }
  
  cat("Training finished in",
      round(as.numeric(difftime(Sys.time(), t0, units = "secs"))),
      "sec\n")
  
  tracker$clean()
  history <- tracker$collection
  list(model = model, history = history)
}

plot_history <- function(history, prefix) {
  if (length(history) == 0) return()
  metrics <- names(history[[1]])
  df <- do.call(rbind, lapply(seq_along(history), function(i) {
    data.frame(epoch = i, t(sapply(history[[i]], as.numeric)))
  }))
  df_long <- tidyr::pivot_longer(df, cols = -epoch,
                                 names_to = "metric", values_to = "value")
  
  p <- ggplot(df_long, aes(x = epoch, y = value, color = metric)) +
    geom_line() +
    theme_minimal() +
    labs(title = "Training History")
  outfile <- paste0(prefix, "history.png")
  ggsave(outfile, p, width = 8, height = 8, dpi = 300)
  cat(outfile, "\n")
}
