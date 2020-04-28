data {
  int<lower=0> N;
  vector[N] male;
  vector[N] age;
  vector[N] eth;
  vector[N] income;
}
parameters {
  real alpha;
  real beta1;
  real beta2;
  real beta3;
  real beta4;
}
model {
  service_failure ~ bernoulli_logit(alpha + beta1 * male + beta2 * age + beta3 * eth + beta4 * income);
}

