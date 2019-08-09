# Stan/Rethinking Setup ---------------------------------------------------
# 1. Install Stan, a compiler, and rstan: github.com/stan-dev/rstan/wiki/RStan-Getting-Started
# 2. Install rethinking package: https://github.com/rmcelreath/rethinking/tree/Experimental

# Simulating Data ---------------------------------------------------------
# Load packages.
library(rethinking)
library(rstan)

# Simulate explanatory variables.
sim_values = list(
  N = 1000,
  length <- sample(10:280, size = 1000 , replace = TRUE),
  past <- sample(0:1 , size = 1000 , replace = TRUE),  
  retweet <- runif(1000, min=0, max=5),    
  followers <- sample(0:1000, size=1000, replace=TRUE)
)

R <- 50

# Specify parameter values.
b0 <- 1
bl <- 5
bp <- 10
br <- -3
bf <- 2
error <- rnorm(1000, 0, 1) 

# Specify the number of draws (i.e., simulated datasets).
R <- 50

# Simulate data.
sim_data <- stan(
  file = here::here("Code","stan_ecology_model.stan"), 
  data = sim_values,
  iter = R,
  warmup = 0, 
  chains = 1, 
  refresh = R,
  seed = 42,
  algorithm = "Fixed_param"
)
