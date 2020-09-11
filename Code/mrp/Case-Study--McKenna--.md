Case Study (McKenna)
================
McKenna Weech, Hannah Thompson
3/10/2020

Multi-level regression and post-stratification helps to infer treatment
size for a whole population from a non-representative sample. Using
population knowledge sample are classified into cells based off of
chosen classification variables. Hierarchical Bayesian models are used
to infer(?) treatment within cells and population knowledge is used to
weight each cell according to their prevalence within the total
population.

## Data

Categorical classification variables are chosen and cell are assigned
based off of category classification. For example, if one variable is
gender consisting of two categories (male and other) and one variable is
age consisting of two categories (0-50 and 50-100) then there would be
four cells: male 0-50, male 50-100, other 0-50, and other 50-100.

In this case study there are five classification variables chosen: age,
income, ethnicity, gender, and state. Each variable is transformed into
categorical variables with age having seven categories, income having
three, ethnicity having three, gender having two, and state have 50
making the total number of cells 6,300 (7 x 3 x 3 x 2 x 50 = 6300).

Data is generated using a function from…(find reference).

## Model

This case study will use generated data. The data is broken up into
three data tables: sample, post-stratification, and population. The
post-stratification table shows the population density of each cell and
is used to weight the outcome results from the sample. Because the data
is generated we also have a generated population that we can use to
check the accuracy of the model.

First we will build a model to predict the our outcome for each cell.
Then we will use the post-stratification table to correctly weight each
cell to predict outcome for the whole population.

We will use Bayesian hierarchical modeling to take advantage of partial
pooling to account for empty cells.

### McKenna’s logit model

``` r
predictors <- sample %>% 
  select(male, age, eth, income, state)

data_list <- list(
  D = 5,                                              # Number of variables. 
  N = 1200,                                           # Number of observations.
  L = 5,                                              # Number of groups. 
  y = sample$service_failure,                         # Outcome variables. 
  ll = sample(5, 1200, replace = TRUE),               # Group assignment. 
  x = predictors                                      # Matrix of predictors. 
)
```

``` r
fit <- stan(
  file = here::here("Code", "mrp", "logit_hier.stan"),
  data = data_list,
  seed = 42 
)

fit
```

``` r
posterior_prob <- posterior_linpred(fit, transform = TRUE, newdata = poststrat)
poststrat_prob <- posterior_prob %*% poststrat$N / sum(poststrat$N)
model_popn_pref <- c(mean = mean(poststrat_prob), sd = sd(poststrat_prob))
round(model_popn_pref, 3)

true_popn_pref <- sum(true_popn$satisfaction * poststrat$N) / sum(poststrat$N)
round(true_popn_pref, 3)

state_df <- data.frame(
  State = 1:50,
  model_state_sd = rep(-1, 50),
  model_state_pref = rep(-1, 50),
  sample_state_pref = rep(-1, 50),
  true_state_pref = rep(-1, 50),
  N = rep(-1, 50)
)

for(i in 1:length(levels(as.factor(poststrat$state)))) {
  poststrat_state <- poststrat[poststrat$state == i, ]
    posterior_prob_state <- posterior_linpred(
    fit,
    transform = TRUE,
    draws = 1000,
    newdata = as.data.frame(poststrat_state)
  )
  poststrat_prob_state <- (posterior_prob_state %*% poststrat_state$N) / sum(poststrat_state$N)
  #This is the estimate for popn in state:
  state_df$model_state_pref[i] <- round(mean(poststrat_prob_state), 4)
  state_df$model_state_sd[i] <- round(sd(poststrat_prob_state), 4)
  #This is the estimate for sample
  state_df$sample_state_pref[i] <- round(mean(sample$cat_pref[sample$state == i]), 4)
  #And what is the actual popn?
  state_df$true_state_pref[i] <-
    round(sum(true_popn$cat_pref[true_popn$state == i] * poststrat_state$N) /
            sum(poststrat_state$N), digits = 4)
  state_df$N[i] <- length(sample$satisfaction[sample$state == i])
}

state_df[c(1,3:6)]
```

#### Downes logit model

``` r
# Observations. 
downes_list <- list(
  outcome = sample$service_failure,
  male = sample$male + 1,
  age = sample$age, 
  eth = sample$eth, 
  income = sample$income,
  state = sample$state,
  n = length(sample$service_failure), 
  n_male = max(sample$male) + 1,
  n_age = max(sample$age),
  n_eth = max(sample$eth),
  n_income = max(sample$income),
  n_state = max(sample$state)
)

  n_iter <- 1000
  n_chain <- 1
  n_sim <- (n_iter * n_chain)/2 
```

``` r
fit <- stan(
  file = here::here("Code", "mrp", "downes_copy.stan"),
  data = downes_list,
  seed = 42 
)
```

``` r
  stanfit_final_sim <- extract(fit, pars=c("b0","a_income","a_age","a_eth","a_state","a_male","sigma_age","sigma_eth","sigma_income", "sigma_male","sigma_state"))

n_pscell <- length(poststrat$N)
n_state <- max(poststrat$state)

stanfit_final_ypred <- array(NA, c(n_sim, n_pscell))
  for(l in 1:n_pscell){
    stanfit_final_ypred[,l] <- invlogit(stanfit_final_sim$b0 + 
                                        stanfit_final_sim$a_age * poststrat$age[l] +
                                        stanfit_final_sim$a_income * poststrat$income[l] +
                                        stanfit_final_sim$a_eth * poststrat$eth[l] +
                                        stanfit_final_sim$a_state * poststrat$state[l] +
                                        stanfit_final_sim$a_male * poststrat$male[l])
  }  

  stanfit_final_ps <- c(NA, n_sim)
  for(s in 1:n_sim){
    stanfit_final_ps[s] <- sum(poststrat$N*stanfit_final_ypred[s,])/sum(poststrat$N)
  }
  
stanfit_final_q <- round(quantile(stanfit_final_ps, c(0.025,0.50,0.975)),3)
  stanfit_final_sd <- round(sd(stanfit_final_ps),4)
  stanfit_final_pop_est <- cbind(stanfit_final_q[2], stanfit_final_sd, stanfit_final_q[1],
                                 stanfit_final_q[3])  
```

#### Stan user guide logit model

``` r
stan_data <- list( 
  age = sample$age, 
  eth = sample$eth, 
  income = sample$income, 
  male = sample$male, 
  state = sample$state, 
  Y = sample$service_failure, 
  N = length(sample$service_failure)
)
```

``` r
fit <- stan(
  file = here::here("Code", "mrp", "stan_mrp.stan"),
  data = stan_data,
  seed = 42 
)
```

## Post-Stratification

Post-stratification consists of taking the sum of the estimates of the
model for each cell times the number of people in each cell divided by
the total number of
people

\[\omega=\displaystyle\frac{\]*{j=1}^{J}*{j}\*N\_{j}\[}{\]*{j=1}^{J}N*{j}\[}\]
Post-stratification can be performed in Stan using the generated
quantities block.

``` r
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
```

## Compare to Population

Because we are using simulated data we can see how our Post-Stratified
predictions compare to the true population.
