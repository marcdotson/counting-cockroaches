data {
  int<lower=0> N;
  vector[N] x;
  int<lower=0,upper=1> y[N];
}
parameters {
  vector[N] alpha;
  real beta;
}
model {
  y ~ bernoulli_logit(alpha[N] + alpha[N] + alpha[N] + alpha[N] + beta * x);
}

