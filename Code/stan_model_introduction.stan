data {
  int N; 
  real Y[N];
}

parameters {
  real mu;
  real<lower=0> sigma;
}

model {
  for(i in 1:N)
    Y[i] ~ normal(mu, sigma);
  mu ~ normal(1.7, 0.3);
  sigma ~ cauchy(0, 1);
}

