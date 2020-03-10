data {
  int<lower=0> N;
  vector[N] male;
  vector[N] age;
  vector[N] eth;
  vector[N] income;
  vector[N] state;
  vector[N] service_failure;
}

parameters {
  real alpha;
  real beta1;
  // real beta2;
  // real beta3;
  // real beta4;
  // real beta5;
  real<lower=0> sigma;
}

model {
  service_failure ~ bernoulli_logit(alpha + beta1 * male, sigma);
}
