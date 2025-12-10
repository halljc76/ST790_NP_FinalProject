# full sim
source("sim_settings.R")
source("one_rep.R")

run_full_simulation <- function() {
  
  outdir <- "simulation_results/"
  dir.create(outdir, showWarnings = FALSE)
  
  results <- list()
  
  for (i in seq_len(nrow(sim_settings))) {
    setting <- sim_settings[i, ]
    
    key <- sprintf("beta%.2f_N%d_V%d", 
                   setting$beta_stn, setting$n_indivs, setting$n_voxels)
    message("\n========== Running setting: ", key, " ==========\n")
    
    setting_dir <- paste0(outdir, key, "/")
    dir.create(setting_dir, showWarnings = FALSE)
    
    # store 20 reps
    reps <- vector("list", 20)
    for (rep in 1:20) {
      reps[[rep]] <- run_one_replicate(
        beta_stn = setting$beta_stn,
        n_indivs = setting$n_indivs,
        n_voxels = setting$n_voxels,
        rep_id = rep,
        outdir = setting_dir
      )
    }
    
    results[[key]] <- reps
    saveRDS(results[[key]], paste0(setting_dir, "replicates_raw_results.rds"))
  }
  
  saveRDS(results, paste0(outdir, "all_results.rds"))
  return(results)
}

# Run everything:
NoSleepR::nosleep_on(keep_display = TRUE)
final_results <- run_full_simulation()
NoSleepR::nosleep_off()