source("viz.R")

plots <- plot_replicate_results("simulation_results/beta0.10_N20_V64", rep_id = 1)

for (name in names(plots)) {
  ggsave(
    filename = paste0("simulation_results/plots/", name, ".png"),
    plot = plots[[name]],
    width = 6, height = 6
  )
}

for (p in plots) print(p)
