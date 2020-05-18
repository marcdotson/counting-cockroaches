data {
  int<lower=0> n;                                       // Number of survey participants
  
  int<lower=0> n_age5p;                                 // These are the number of categories
  int<lower=0> n_ra;                                    // for each poststratification factor
  int<lower=0> n_ingp;                                  // in the final model
  int<lower=0> n_ieo;                                   
  int<lower=0> n_englp;                                 
  int<lower=0> n_occp;                                   
  
  int<lower=0,upper=1> outcome[n];                      // observed outcome data, 1=Yes, 0=No
  
  int<lower=0,upper=n_age5p> age5p[n];                  // These are the variables containing
  int<lower=0,upper=n_ra> ra[n];                        // the observed data for each
  int<lower=0,upper=n_ingp> ingp[n];                    // poststratification factor in the 
  int<lower=0,upper=n_ieo> ieo[n];                      // final model
  int<lower=0,upper=n_englp> englp[n];                  
  int<lower=0,upper=n_occp> occp[n];                    
 }
parameters {                                            // Model parameters to be estimated:
  real b0;                                              // "intercept"
  real b_age5p;                                         // fixed "slope" for age5p (ordinal)
  real b_ingp;                                          // fixed (binary) effect ingp
  real b_ieo;                                           // fixed "slope" for seifa ieo (ordinal)
  real a_age5p[n_age5p];                                // age varying coeffs (depart from linear fit)
  real a_ra[n_ra];                                      // remoteness class varying coeffs 
  real a_ieo[n_ieo];                                    // seifa ieo varying coeffs (depart linear fit)
  real a_englp[n_englp];                                // english fluency varying coeffs
  real a_occp[n_occp];                                  // occupation varying coeffs
  real<lower=0> sigma_age5p;                            // age var(sd) component
  real<lower=0> sigma_ra;                               // remoteness class var(sd) component
  real<lower=0> sigma_ieo;                              // seifa ieo var(sd) component
  real<lower=0> sigma_englp;                            // english fluency var(sd) component
  real<lower=0> sigma_occp;                             // occupation var(sd) component
}
transformed parameters {                                // Model specification is here
  vector[n] outcome_hat;
  for (i in 1:n)
    outcome_hat[i] = b0 + b_age5p* age5p[i] + 
                           b_ingp * ingp[i] +
                           b_ieo * ieo[i] +
                           a_age5p[age5p[i]] + 
                           a_ra[ra[i]] +
                           a_ieo[ieo[i]] +
                           a_englp[englp[i]] +
                           a_occp[occp[i]];
}
model {                                                 // Model distributions for varying coefficients
  b0 ~ cauchy(0,2.5);                                   // and prior distributions for unmodelled
  b_age5p ~ cauchy(0,2.5);                              // parameters are specified here
  b_ingp ~ cauchy(0,2.5);
  b_ieo ~ cauchy(0,2.5);                                // Can modify Cauchy scale parameter
                                                        // Can also specify a uniform distribution here
  a_age5p ~ normal(0, sigma_age5p);                     // or by setting upper and lower limits for
  a_ra ~ normal(0, sigma_ra);                           // unmodelled parameters in the parameter step
  a_ieo ~ normal(0, sigma_ieo);                         // above  
  a_englp ~ normal(0, sigma_englp);                      
  a_occp ~ normal(0, sigma_occp);
  
  sigma_age5p ~ cauchy(0,2.5);
  sigma_ra ~ cauchy(0,2.5);
  sigma_ieo ~ cauchy(0,2.5);
  sigma_englp ~ cauchy(0,2.5);
  sigma_occp ~ cauchy(0,2.5);
  
  outcome ~ bernoulli_logit(outcome_hat);               // For a continuous outcome, specify a
                                                        // normal distribution with an additional
                                                        // variance component, i.e.
                                                        // outcome ~ normal(outcome_hat, sigma_outcome)
                                                        // Note the prior distributions for 
                                                        // unmodelled parameters will change.
}

