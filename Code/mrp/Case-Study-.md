MRP Case Study
================
McKenna Weech
1/17/2020

Load packages

``` r
library(rstanarm)
library(ggplot2)
library(bayesplot)
theme_set(bayesplot::theme_default())
library(dplyr)
library(tidyr)
```

First we will simulate data.

``` r
simulate_mrp_data <- function(n) {
  J <- c(2, 3, 7, 3, 50) # male or not, eth, age, income level, state
  poststrat <- as.data.frame(array(NA, c(prod(J), length(J)+1))) # Columns of post-strat matrix, plus one for size
  colnames(poststrat) <- c("male", "eth", "age","income", "state",'N')
  count <- 0
  for (i1 in 1:J[1]){
    for (i2 in 1:J[2]){
      for (i3 in 1:J[3]){
        for (i4 in 1:J[4]){
          for (i5 in 1:J[5]){
              count <- count + 1
              # Fill them in so we know what category we are referring to
              poststrat[count, 1:5] <- c(i1-1, i2, i3, i4, i5) 
          }
        }
      }
    }
  }
  # Proportion in each sample in the population
  p_male <- c(0.52, 0.48)
  p_eth <- c(0.5, 0.2, 0.3)
  p_age <- c(0.2,.1,0.2,0.2, 0.10, 0.1, 0.1)
  p_income<-c(.50,.35,.15)
  p_state_tmp<-runif(50,10,20)
  p_state<-p_state_tmp/sum(p_state_tmp)
  poststrat$N<-0
  for (j in 1:prod(J)){
    poststrat$N[j] <- round(250e6 * p_male[poststrat[j,1]+1] * p_eth[poststrat[j,2]] *
      p_age[poststrat[j,3]]*p_income[poststrat[j,4]]*p_state[poststrat[j,5]]) #Adjust the N to be the number observed in each category in each group
  }
  
  # Now let's adjust for the probability of response
  p_response_baseline <- 0.01
  p_response_male <- c(2, 0.8) / 2.8
  p_response_eth <- c(1, 1.2, 2.5) / 4.7
  p_response_age <- c(1, 0.4, 1, 1.5,  3, 5, 7) / 18.9
  p_response_inc <- c(1, 0.9, 0.8) / 2.7
  p_response_state <- rbeta(50, 1, 1)
  p_response_state <- p_response_state / sum(p_response_state)
  p_response <- rep(NA, prod(J))
  for (j in 1:prod(J)) {
    p_response[j] <-
      p_response_baseline * p_response_male[poststrat[j, 1] + 1] *
      p_response_eth[poststrat[j, 2]] * p_response_age[poststrat[j, 3]] *
      p_response_inc[poststrat[j, 4]] * p_response_state[poststrat[j, 5]]
  }
  people <- sample(prod(J), n, replace = TRUE, prob = poststrat$N * p_response)
  
  ## For respondent i, people[i] is that person's poststrat cell,
  ## some number between 1 and 32
  n_cell <- rep(NA, prod(J))
  for (j in 1:prod(J)) {
    n_cell[j] <- sum(people == j)
  }
  
  coef_male <- c(0,-0.3)
  coef_eth <- c(0, 0.6, 0.9)
  coef_age <- c(0,-0.2,-0.3, 0.4, 0.5, 0.7, 0.8, 0.9)
  coef_income <- c(0,-0.2, 0.6)
  coef_state <- c(0, round(rnorm(49, 0, 1), 1))
  coef_age_male <- t(cbind(c(0, .1, .23, .3, .43, .5, .6),
                           c(0, -.1, -.23, -.5, -.43, -.5, -.6)))
  true_popn <- data.frame(poststrat[, 1:5], service_failure = rep(NA, prod(J)))
  for (j in 1:prod(J)) {
    true_popn$cat_pref[j] <- plogis(
      coef_male[poststrat[j, 1] + 1] +
        coef_eth[poststrat[j, 2]] + coef_age[poststrat[j, 3]] +
        coef_income[poststrat[j, 4]] + coef_state[poststrat[j, 5]] +
        coef_age_male[poststrat[j, 1] + 1, poststrat[j, 3]]
      )
  }
  
  #male or not, eth, age, income level, state, city
  y <- rbinom(n, 1, true_popn$cat_pref[people])
  male <- poststrat[people, 1]
  eth <- poststrat[people, 2]
  age <- poststrat[people, 3]
  income <- poststrat[people, 4]
  state <- poststrat[people, 5]
  
  sample <- data.frame(service_failure = y, 
                       male, age, eth, income, state, 
                       id = 1:length(people))
  
  #Make all numeric:
  for (i in 1:ncol(poststrat)) {
    poststrat[, i] <- as.numeric(poststrat[, i])
  }
  for (i in 1:ncol(true_popn)) {
    true_popn[, i] <- as.numeric(true_popn[, i])
  }
  for (i in 1:ncol(sample)) {
    sample[, i] <- as.numeric(sample[, i])
  }
  list(
    sample = sample,
    poststrat = poststrat,
    true_popn = true_popn
  )
}
```

Next we can view the data

``` r
mrp_sim <- simulate_mrp_data(n=100)
str(mrp_sim)
```

    ## List of 3
    ##  $ sample   :'data.frame':   100 obs. of  7 variables:
    ##   ..$ service_failure: num [1:100] 0 1 1 1 1 1 0 0 1 1 ...
    ##   ..$ male           : num [1:100] 1 0 0 0 0 0 0 0 0 0 ...
    ##   ..$ age            : num [1:100] 1 7 4 4 7 4 6 4 7 7 ...
    ##   ..$ eth            : num [1:100] 1 1 3 1 3 3 3 3 1 1 ...
    ##   ..$ income         : num [1:100] 2 2 1 1 1 1 2 1 3 1 ...
    ##   ..$ state          : num [1:100] 12 15 25 47 14 39 15 50 4 28 ...
    ##   ..$ id             : num [1:100] 1 2 3 4 5 6 7 8 9 10 ...
    ##  $ poststrat:'data.frame':   6300 obs. of  6 variables:
    ##   ..$ male  : num [1:6300] 0 0 0 0 0 0 0 0 0 0 ...
    ##   ..$ eth   : num [1:6300] 1 1 1 1 1 1 1 1 1 1 ...
    ##   ..$ age   : num [1:6300] 1 1 1 1 1 1 1 1 1 1 ...
    ##   ..$ income: num [1:6300] 1 1 1 1 1 1 1 1 1 1 ...
    ##   ..$ state : num [1:6300] 1 2 3 4 5 6 7 8 9 10 ...
    ##   ..$ N     : num [1:6300] 133801 130527 138244 171782 159145 ...
    ##  $ true_popn:'data.frame':   6300 obs. of  7 variables:
    ##   ..$ male           : num [1:6300] 0 0 0 0 0 0 0 0 0 0 ...
    ##   ..$ eth            : num [1:6300] 1 1 1 1 1 1 1 1 1 1 ...
    ##   ..$ age            : num [1:6300] 1 1 1 1 1 1 1 1 1 1 ...
    ##   ..$ income         : num [1:6300] 1 1 1 1 1 1 1 1 1 1 ...
    ##   ..$ state          : num [1:6300] 1 2 3 4 5 6 7 8 9 10 ...
    ##   ..$ service_failure: num [1:6300] NA NA NA NA NA NA NA NA NA NA ...
    ##   ..$ cat_pref       : num [1:6300] 0.5 0.109 0.154 0.45 0.198 ...

``` r
sample <- mrp_sim[["sample"]]
poststrat <- mrp_sim[["poststrat"]]
true_popn <- mrp_sim[["true_popn"]]
```

### Begine Bayseian Work Flow (Modeling)

We are woking towards building a model that can predict the severity of
a service failure baised on consumer feedback. The case study will work
to build a simple model and impliment postratification to generalize
findings in a sample to a larger population. The model we will focus on
here will start as a simple binomial model but we will hopefully build
it up. We will be using code taken from Lauren Kenndey and Jonah Gabry’s
vignette on postratification see:
<http://mc-stan.org/rstanarm/articles/mrp.html>

We will be lookig at the following pices of data about each “customer”

  - gender
  - age (group)
  - income (group)
  - state

All of the variables are categorical variables.

WE NEED A MODEL

I HATE RSTANARM – JUST SAYING

### Postratification

Note: for now I have set these chuncks to eval = false because we don t
have a poststrat\_prob yet so it freaks out

For the postratification portion we take the estimate that we get from
the model times poststrat\(/N / sum(postsrat\)N)

``` r
poststrat_prob <- posterior_prob %*% poststrat$N / sum(poststrat$N)
model_popn_pref <- c(mean = mean(poststrat_prob), sd = sd(poststrat_prob))
round(model_popn_pref, 3)
```

Becasue we are using simulated data we can see how our poststratified
predictions compare to the true population

``` r
sample_popn_pref <- mean(sample$cat_pref)
round(sample_popn_pref, 3)

compare2 <- compare2 +
  geom_hline(yintercept = model_popn_pref[1], colour = '#2ca25f', size = 1) +
  geom_text(aes(x = 5.2, y = model_popn_pref[1] + .025), label = "MRP", colour = '#2ca25f')
bayesplot_grid(compare, compare2, 
               grid_args = list(nrow = 1, widths = c(8, 1)))
```