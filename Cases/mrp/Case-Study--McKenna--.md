Case Study (McKenna)
================
McKenna Weech
3/10/2020

## Intro

Multi-level regression and post-stratification helps to infer treatment
size for a whole population from a non-representative sample. Using
population knowlege sample are classified into cells baised off of
choosen classification variables. Heirarchical bayesian models are used
to infer(?) treatment within cells and population knowlege is used to
weight each cell according to their prevelance within the total
population.

## Data

Categorical classification variables are choosen and cell are assigned
baised off of category classification. For example, if one variable is
gender consisting of two categories (male and other) and one varibale is
age consisting of two categories (0-50 and 50-100) then there would be
four cells: male 0-50, male 50-100, other 0-50, and other 50-100.

In this case study there are five classification variables choosen: age,
income, ethnicity, gender, and state. Each variable is tranforemed into
categorical variables with age having seven categories, income having
three, ethnicity having three, gender having two, and state have 50
making the total number of cells 6,300 (7 x 3 x 3 x 2 x 50 = 6300).

Data is generated using a function from…(find referance)

## Model

This case study will use generated data. The data is broken up into
three data tables: sample, postratification, and population. The
postratification table shows the population density of each cell and is
used to weight the outcome results from the sample. Because the data is
generated we also have a generated population that we can use to check
the accuracy of the model.

First we will build a model to predict the our outcome for each cell.
Then we will use the postratification table to correctly weight each
cell to predict outcome for the whole population.

We will use bayesian hierchical modeling to take advantage of partical
pooling to account for empty cells.

#### McKenna’s logit model

  - The issue here is the way that the matrix of predictors is being
    specified, I think

<!-- end list -->

``` r
predictors <- sample %>% 
  select(male, age, eth, income, state)

pred <- as.matrix(predictors)

# pred <- list()
# for (n in 1:nrow(predictors)) {
#   pred[[n]] <- matrix(as.numeric(predictors[n,]))
# }
```

``` r
data_list <- list(
  D = 5,                                              # Number of variables. 
  N = 1200,                                           # Number of observations.
  L = 5,                                              # Number of groups. 
  y = sample$service_failure,                         # Outcome variables. 
  ll = sample(5, 1200, replace = TRUE),               # Group assignment. 
  x = pred                                            # Matrix of predictors. 
)
```

``` r
fit <- stan(
  file = here::here("Code", "mrp", "logit_hier.stan"),
  data = "data_list",
  seed = 42 
)
```

#### Downes logit model

``` r
# Observations. 
downes_list <- list(
  outcome = sample$service_failure,
  male = sample$male,
  age = sample$age, 
  eth = sample$eth, 
  income = sample$income,
  state = sample$state,
  n = length(sample$service_failure), 
  n_male = max(sample$male),
  n_age = max(sample$age),
  n_eth = max(sample$eth),
  n_income = max(sample$income),
  n_state = max(sample$state)
)

# Define parameters for Hamiltonian Monte Carlo algorithm
n_iter <- 1000
n_chain <- 4
n_sim <- (n_iter * n_chain)/2 
```

``` r
stanfit_final <- stan(file = "downes_copy.stan",
                        data = downes_list, 
                        iter = n_iter, chains = 1)
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
stanfit_final <- stan(file= "stan_mrp.stan",
                        data= "stan_data", 
                        iter=n_iter, chains=n_chain)
```

## Postratification

Postratification consists of taking the sum of the estimates of the
model for each cell times the number of people in each cell divided by
the total number of
people

\[\omega=\displaystyle\frac{\]*{j=1}^{J}*{j}\*N\_{j}\[}{\]*{j=1}^{J}N*{j}\[}\]
Postratification can be performed in stan using the generated quantities
block.

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

Because we are using simulated data we can see how our poststratified
predictions compare to the true population
