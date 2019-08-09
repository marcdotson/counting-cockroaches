// Observed choices and the experimental design.
data {
  int N;             // Number of observations.
  int Y[N];          // Vector of observed choices.
  real length[N];
  real past[N];
  real retweet[N];
  real followers[N];
  int<lower=0> p;    // Number of parameters
}
  
parameters {
// Define parameters to estimate
  real beta[p];
   
// standard deviation (a positive real number)
  real<lower=0> sigma;
}

transformed parameters  {
  real mu[N];
  for (i in 1:N) {
    mu[i] = beta[1] + beta[2]*length[i] + beta[3]*past[i] + beta[4]*retweet[i] + beta[5]*followers[i]; 
   }
 }

// Multinomial logit model.
model {
  // Standard normal prior for beta.
  beta ~ normal(0, 1);
  mu ~ normal(3,1);
}
