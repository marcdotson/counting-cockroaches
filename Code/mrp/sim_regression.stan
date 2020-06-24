
data {
  int<lower=0> N;
  vector[N] service_failure;
  vector[N] male;
  vector[N] age;
  vector[N] eth;
  vector[N] income;
  vector[N] state;
}


parameters {
  real<lower=0> sigma;
  real alpha;
  real beta1;
  real beta2;
  real beta3;
  real beta4;
  real beta5;
}


model {
  service_failure ~ normal(alpha + beta1 * male + beta2 * age + beta3 * eth + beta4 * income + beta5 * state, sigma);
}



cbind(service_failure, no service failuer) ˜
service_failure +
(1 + male | year) +
year_std + year_sq_std +
(1 + year_std + year_sq_std | state) +
(1 | age) +
(1 | edu) +
(1 | sex_race) +
(1 | age:year) +
(1 | edu:year) +
(1 | sex_race:year) +
(1 | state:year)
