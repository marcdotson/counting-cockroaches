# Load packages.
library(tidyverse)
library(rstan)
library(bayesplot)
library(tidybayes)

# Set Stan options.
options(mc.cores = parallel::detectCores())

# Specify data and hyperparameter values.
sim_values <- list(
  N = 100,                            # Number of individuals.
  K = 2,                              # Number of groups.
  g = sample(2, 100, replace = TRUE), # Vector of group assignments.
  mu = .76,                           # Mean of the population model.
  tau = .1                            # Variance of the population model.
)

# Generate data.
sim_data <- stan(
  file = here::here("Code", "mrp", "generate_data.stan"),
  data = sim_values,
  iter = 1,
  chains = 1,
  seed = 42,
  algorithm = "Fixed_param"
)

# Extract simulated data and group intercepts.
sim_y <- extract(sim_data)$y
sim_beta <- extract(sim_data)$beta

# Calibrate the model.
fit <- stan(
  file = here::here("Code", "mrp", "regression.stan"),
  data = data,
  control = list(adapt_delta = 0.99),
  seed = 42
)

tibble(sim_beta, sim_y)







