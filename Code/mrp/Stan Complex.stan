data {
  int<lower=0> K;               // number of predictors
  int<lower=0> N[2];            // number of observations where y =0 and y = 1 respectively
  vector[K] xbar;               // vector of column-means of rbind(X0, X1)
  int<lower=0,upper=1> dense_X; // flag for dense vs. sparse
  row_vector[D] x[N];
}
parameters {
  real mu[D];
  real<lower=0> sigma[D];
  vector[D] beta[L];
}
model {
  for (d in 1:D) {
    mu[d] ~ normal(0, 100);
    for (l in 1:L)
      beta[l,d] ~ normal(mu[d], sigma[d]);
  }
  for (n in 1:N)
    y[n] ~ bernoulli(inv_logit(x[n] * beta[ll[n]]));
}