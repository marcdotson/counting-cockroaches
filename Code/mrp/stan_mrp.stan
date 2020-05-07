data {
  int<lower = 0> N;
  int<lower = 1, upper = 7> age[N];
  int<lower = 1, upper = 3> income[N];
  int<lower = 1, upper = 3> eth[N];
  int<lower = 1, upper = 2> male[N];
  int<lower = 1, upper = 50> state[N];
  int<lower = 0> Y[N];
  int<lower = 0> P[7, 3, 3, 2, 50];
}
parameters {
  
  real<lower = 0> sigma_beta;
  real<lower = 0> sigma_gamma;
  real<lower = 0> sigma_delta;
  real<lower = 0> sigma_omega;
  real<lower = 0> sigma_rho;
  real epsilon;
  
  real alpha;
  vector<multiplier = sigma_beta>[7] beta;
  vector<multiplier = sigma_gamma>[3] gamma;
  vector<multiplier = sigma_delta>[3] delta;
  vector<multiplier = sigma_omega>[2] omega;
  vector<multiplier = sigma_rho>[50] rho;
  
}
model {
  y ~ bernoulli_logit(alpha + beta[age] + gamma[income] + delta[eth] + omega[male] + rho[state]);
  alpha ~ normal(0, 2);
  beta ~ normal(0, sigma_beta);
  gamma ~ normal(0, sigma_gamma);
  delta ~ normal(0, sigma_delta);
  omega ~ normal(0, sigma_omega);
  rho ~ norma(0, sigma_rho);
  { sigma_beta, sigma_gamma, sigma_delta, sigma_omega, sigma_rho } ~ normal(0, 1);
}
generated quantities {
  real expect_pos = 0;
  int total = 0;
  for (b in 1:7)
    for (c in 1:3)
      for (d in 1:3) 
        for (f in 1:2)
          for (g in 1:50){
        total += P[b, c, d, f, g];
        expect_pos
          += P[b, c, d, f, g]
             * inv_logit(alpha + beta[b] + gamma[c] + delta[d] + omega[f] + rho[g]);
      }
  real<lower = 0, upper = 1> phi = expect_pos / total;
}


