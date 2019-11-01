// Index values and observations.
data {
  int<lower=1> N;             // Number of individuals.
  int<lower=1> K;             // Number of groups.
  vector[N] y;                // Vector of observations.
  int<lower=1, upper=K> g[N]; // Vector of group assignments.
}

// Parameters and hyperparameters.
parameters {
  vector[K] beta;    // Vector of group intercepts.
  real mu;           // Mean of the population model.
  real<lower=0> tau; // Variance of the population model.
}

// Hierarchical regression.
model {
  // Hyperpriors.
  mu ~ normal(0, 5);
  tau ~ cauchy(0, 2.5);

  // Population model and likelihood.
  beta ~ normal(mu, tau);
  for (n in 1:N) {
    y[n] ~ normal(beta[g[n]], 1);
  }
}

