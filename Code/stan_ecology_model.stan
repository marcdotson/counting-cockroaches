// Observed choices and the experimental design.
data {
  int <lower=1> N;
  vector[N] y;
}
  
parameters {
  real mu;
  real<lower=0> sigma;
}

model {
  //Priors
  mu ~ normal(3.5,1);
  sigma ~ cauchy(0,2.5);
}
