# utils.R
# utility functions for data manipulation, evaluation, saving, etc.
library(abind)
library(ggplot2)

# print std error
erint <- function(...) {
  cat(..., file=stderr(), sep=" ", fill=TRUE)
}

# multidimensional slicing
thin_3d <- function(x, start, stop, stride) {
  # Ensure x has 4 dimensions: (batch, x, y, z)
  dims <- dim(x)
  if(length(dims) != 4) stop("x must be 4-dimensional: (batch, X, Y, Z)")
  
  # Create sequences for each spatial dimension
  seq_x <- seq(start[1] + 1, stop[1], by = stride[1])
  seq_y <- seq(start[2] + 1, stop[2], by = stride[2])
  seq_z <- seq(start[3] + 1, stop[3], by = stride[3])
  
  # Subset using all four dimensions explicitly
  x_out <- x[, seq_x, seq_y, seq_z, drop = FALSE]
  return(x_out)
}


# equality check
alleq <- function(y, x) {
  if(class(y) != class(x)) return(FALSE)
  if(is.list(x) || is.vector(x)){
    if(length(y) != length(x)) return(FALSE)
    return(all(mapply(alleq, y, x)))
  } else if(is.matrix(x) || is.array(x)){
    return(all(y == x))
  } else {
    return(y == x)
  }
}

# make directories for saving files
mkdir <- function(path) {
  dir <- dirname(path)
  if(dir != "") dir.create(dir, recursive=TRUE, showWarnings=FALSE)
}

# save/load files
save_pickle <- function(x, filename) {
  mkdir(filename)
  saveRDS(x, filename)
  cat(filename, "saved\n")
}

load_pickle <- function(filename) {
  x <- readRDS(filename)
  cat("Pickle loaded from", filename, "\n")
  return(x)
}

# save image
save_image <- function(img, filename, normalization='negpos', cmap='RdBu') {
  if(normalization != 'none'){
    img <- img
    if(normalization == 'negpos'){
      img <- img / (max(abs(img), na.rm=TRUE) + 1e-12)
      img <- (img + 1)/2
    } else if(normalization == 'zeroone'){
      img <- (img - min(img, na.rm=TRUE)) / (max(img, na.rm=TRUE) + 1e-12)
    }
  }
  mkdir(filename)
  png(filename)
  image(1:dim(img)[1], 1:dim(img)[2], img[,,1], col=terrain.colors(256))
  dev.off()
  cat(filename, "saved\n")
}

#read/write
read_lines <- function(filename) {
  readLines(filename)
}

write_lines <- function(strings, filename) {
  mkdir(filename)
  writeLines(strings, filename)
  cat(filename, "written\n")
}

write_text <- function(obj, filename, append=FALSE){
  mkdir(filename)
  cat(obj, file=filename, append=append, sep="\n")
  cat(filename, "written\n")
}

# generate unique save ID
gen_saveid <- function(){
  timenow <- format(Sys.time(), "%Y%m%d%H%M%S")
  randstr <- paste0(sample(LETTERS,6,replace=TRUE), collapse="")
  return(paste0(timenow, randstr))
}
