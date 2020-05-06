data {
  int<lower=0> n;                                       // Number of survey participants
  
  int<lower=0> n_age;                                 // These are the number of categories
  int<lower=0> n_eth;                                    // for each poststratification factor
  int<lower=0> n_income;                                  // in the final model
  int<lower=0> n_male;                                   
  int<lower=0> n_state;                                 
  
  int<lower=0,upper=1> outcome[n];                      // observed outcome data, 1=Yes, 0=No
  
  int<lower=0,upper=n_age> age[n];                  // These are the variables containing
  int<lower=0,upper=n_eth> eth[n];                        // the observed data for each
  int<lower=0,upper=n_income> income[n];                    // poststratification factor in the 
  int<lower=0,upper=n_male> male[n];                      // final model
  int<lower=0,upper=n_state> state[n];      
  int<lower = 0> P[7, 3, 3, 2, 50];
 }
parameters {                                            // Model parameters to be estimated:
  real b0;                                              // "intercept"
  real a_age[n_age];                                // age varying coeffs (depart from linear fit)
  real a_eth[n_eth];                                      // remoteness class varying coeffs 
  real a_income[n_income];                                    // seifa ieo varying coeffs (depart linear fit)
  real a_male[n_male];                                // english fluency varying coeffs
  real a_state[n_state];                                  // occupation varying coeffs
  real<lower=0> sigma_age;                            // age var(sd) component
  real<lower=0> sigma_eth;                               // remoteness class var(sd) component
  real<lower=0> sigma_income;                              // seifa ieo var(sd) component
  real<lower=0> sigma_male;                            // english fluency var(sd) component
  real<lower=0> sigma_state;                             // occupation var(sd) component
}
transformed parameters {                                // Model specification is here
  vector[n] outcome_hat;
  for (i in 1:n)
    outcome_hat[i] <- b0 +
                           a_age[age[i]] + 
                           a_eth[eth[i]] +
                           a_income[income[i]] +
                           a_male[male[i]] +
                           a_state[state[i]];
}
model {                                                 // Model distributions for varying coefficients
  b0 ~ cauchy(0,2.5);                                   // and prior distributions for unmodelled                            // Can modify Cauchy scale parameter
                                                        // Can also specify a uniform distribution here
  a_age ~ normal(0, sigma_age);                     // or by setting upper and lower limits for
  a_eth ~ normal(0, sigma_eth);                           // unmodelled parameters in the parameter step
  a_income ~ normal(0, sigma_income);                         // above  
  a_male ~ normal(0, sigma_male);                      
  a_state ~ normal(0, sigma_state);
  
  sigma_age ~ cauchy(0,2.5);
  sigma_eth ~ cauchy(0,2.5);
  sigma_income ~ cauchy(0,2.5);
  sigma_male ~ cauchy(0,2.5);
  sigma_state ~ cauchy(0,2.5);
  
  outcome ~ bernoulli_logit(outcome_hat);               // For a continuous outcome, specify a
                                                        // normal distribution with an additional
                                                        // variance component, i.e.
                                                        // outcome ~ normal(outcome_hat, sigma_outcome)
                                                        // Note the prior distributions for 
                                                        // unmodelled parameters will change.
}
generated quantities {
  real expect_pos = 0;
  int total = 0;
  for (b in 1:7)
    for (c in 1:3)
      for (d in 1:3)
        for (f in 1:2)
          for (g in 1:50) {
        total += P[b, c, d, f, g];
        expect_pos
          += P[b, c, d, f, g]
             * inv_logit(b0 + a_age[b] + a_eth[c] + a_income[d] + a_male[f] + a_state[g]);
      }
  real<lower = 0, upper = 1> phi = expect_pos / total;
}           
