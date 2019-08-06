// Number of observations, choices, etc. to simulate.
data {
  int N;             // Number of observations.
  int P;             // Number of product alternatives.
  int L;             // Number of (estimable) attribute levels.
}

// Simulate data according to the multinomial logit model.
generated quantities {
  int Y[N];          // Vector of observed choices.
  matrix[P, L] X[N]; // Experimental design for each observations.
  vector[L] beta;    // Vector of aggregate beta coefficients.

  // Draw parameter values from the prior.
  for (l in 1:L) {
    beta[l] = normal_rng(0, 1);
  }

  // Generate an experimental design and draw data from the likelihood.
  for (n in 1:N) {
    for (p in 1:P) {
      for (l in 1:L) {
        X[n][p, l] = binomial_rng(1, 0.5);
      }
    }
    Y[n] = categorical_logit_rng(X[n] * beta);
  }
}
