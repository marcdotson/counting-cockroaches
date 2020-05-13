data {
  int<lower = 0> N;
  int<lower = 1, upper = 7> age[N];
  int<lower = 1, upper = 3> income[N];
  int<lower = 1, upper = 3> eth[N];
  int<lower = 1, upper = 2> male[N];
  int<lower = 1, upper = 50> state[N];
  int<lower = 0> y[N];
}
parameters {
  real alpha;
  real epsilon;
  real<lower = 0> sigma_beta;
  real<lower = 0> sigma_gamma;
  real<lower = 0> sigma_delta;
  real<lower = 0> sigma_rho;
  real beta;
  real gamma;
  real delta;
  real rho;
}
model {
  y ~ bernoulli_logit(alpha + beta[age] + gamma[income] + delta[eth] + {epsilon, -epsilon}[male] + rho[state]);
  alpha ~ normal(0, 2);
  epsilon ~ normal(0, 2);
  sigma_beta ~ normal(0, 1);
  sigma_gamma  ~ normal(0, 1);
  sigma_delta ~ normal(0, 1);
  sigma_rho ~ normal(0, 1);
  beta ~ normal(0, sigma_beta);
  gamma ~ normal(0, sigma_gamma);
  delta ~ normal(0, sigma_delta);
  rho ~ norma(0, sigma_rho);
}



