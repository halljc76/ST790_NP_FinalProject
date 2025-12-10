# metrics.R
# Goal: evaluation metrics for image response regression

library(pROC) # for ROC and AUC

mean_squared_difference <- function(true, pred) {
  mean((true - pred)^2)
}

# True positive rate (TPR)
true_positive <- function(true, pred) {
  true_bin <- !isTRUE(all.equal(true, 0)) & !is.na(true) & !is.nan(true)
  true_bin <- abs(true) > 1e-12
  pred_bin <- abs(pred) > 1e-12
  if (sum(true_bin) == 0) return(NA)
  mean(pred_bin[true_bin])
}

# True negative rate (TNR)
true_negative <- function(true, pred) {
  true_bin <- abs(true) < 1e-12
  pred_bin <- abs(pred) < 1e-12
  if (sum(true_bin) == 0) return(NA)
  mean(pred_bin[true_bin])
}

# True discovery (precision)
true_discovery <- function(true, pred) {
  true_bin <- abs(true) > 1e-12
  pred_bin <- abs(pred) > 1e-12
  if (sum(pred_bin) == 0) return(NA)
  mean(true_bin[pred_bin])
}

# True omission (negative predictive value)
true_omission <- function(true, pred) {
  true_bin <- abs(true) < 1e-12
  pred_bin <- abs(pred) < 1e-12
  if (sum(pred_bin) == 0) return(NA)
  mean(true_bin[pred_bin])
}

# False positive rate
false_positive <- function(true, pred) {
  1 - true_negative(true, pred)
}

# False negative rate
false_negative <- function(true, pred) {
  1 - true_positive(true, pred)
}

# False discovery rate
false_discovery <- function(true, pred) {
  1 - true_discovery(true, pred)
}

# False omission rate
false_omission <- function(true, pred) {
  1 - true_omission(true, pred)
}

# ROC AUC
rocauc <- function(true, pred) {
  true_bin <- abs(true) > 1e-12
  pred_val <- abs(pred)
  roc_obj <- roc(true_bin, pred_val)
  as.numeric(auc(roc_obj))
}

# Pearson correlation
correlation_pearson <- function(true, pred) {
  cor(true, pred, method="pearson")
}

# Spearman correlation
correlation_spearman <- function(true, pred) {
  cor(true, pred, method="spearman")
}

# Kendall correlation
correlation_kendall <- function(true, pred) {
  cor(true, pred, method="kendall")
}
