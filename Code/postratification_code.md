Postratification Code
================

## Postratification code

We are looking into an MRP model for Counting Cockroaches and we want to
learn what postratification is and if we can do it in our model.

The documents we study are currently together on a google
    doc

``` r
library(tidyverse)
```

    ## -- Attaching packages ------------------------------------------------ tidyverse 1.2.1 --

    ## v ggplot2 3.2.0     v purrr   0.3.2
    ## v tibble  2.1.3     v dplyr   0.8.3
    ## v tidyr   0.8.3     v stringr 1.4.0
    ## v readr   1.3.1     v forcats 0.4.0

    ## -- Conflicts --------------------------------------------------- tidyverse_conflicts() --
    ## x dplyr::filter() masks stats::filter()
    ## x dplyr::lag()    masks stats::lag()

``` r
vote_yes <- c(rep(0, 25*(1-0.76)), 
              rep(1, 25*0.76),
              rep(0, 50*(1-0.3)),
              rep(1, 50*0.3))
vote_yes
```

    ##  [1] 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0
    ## [36] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1
    ## [71] 1 1 1 1 1

``` r
mean(vote_yes)
```

    ## [1] 0.4533333

That is our population. We will now make a sample with subgroups.

``` r
poll_data <- tibble(
  group = c(rep("A", 25), rep("B", 10)),
  yes = c(rep(1, 19), rep(0, 6), rep(1, 3), rep(0, 7))
)
poll_data %>%
  group_by(group) %>%
  summarise(count = n())
```

    ## # A tibble: 2 x 2
    ##   group count
    ##   <chr> <int>
    ## 1 A        25
    ## 2 B        10

Here are our subgroups and how many our represented in our sample. It
does not match the proportions found in our population.

``` r
Census <- tibble(
  group = c("A", "B"),
  pop = c(25, 50)
)
Census
```

    ## # A tibble: 2 x 2
    ##   group   pop
    ##   <chr> <dbl>
    ## 1 A        25
    ## 2 B        50

``` r
mean(poll_data$yes)
```

    ## [1] 0.6285714

Our mean for the sample varies from our population, so we stratify.

``` r
group_support <- poll_data %>%
  group_by(group) %>%
  summarise(perc_support = mean(yes))
group_support
```

    ## # A tibble: 2 x 2
    ##   group perc_support
    ##   <chr>        <dbl>
    ## 1 A             0.76
    ## 2 B             0.3

``` r
overall_support <- group_support %>%
  summarise(total_support = sum(perc_support * Census$pop/sum(Census$pop)))
overall_support
```

    ## # A tibble: 1 x 1
    ##   total_support
    ##           <dbl>
    ## 1         0.453

``` r
library(rstanarm)
```

    ## Loading required package: Rcpp

    ## rstanarm (Version 2.19.2, packaged: 2019-10-01 20:20:33 UTC)

    ## - Do not expect the default priors to remain the same in future rstanarm versions.

    ## Thus, R scripts should specify priors explicitly, even if they are just the defaults.

    ## - For execution on a local, multicore CPU with excess RAM we recommend calling

    ## options(mc.cores = parallel::detectCores())

    ## - bayesplot theme set to bayesplot::theme_default()

    ##    * Does _not_ affect other ggplot2 plots

    ##    * See ?bayesplot_theme_set for details on theme setting

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
              poststrat[count, 1:5] <- c(i1-1, i2, i3,i4,i5) 
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
  true_popn <- data.frame(poststrat[, 1:5], cat_pref = rep(NA, prod(J)))
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
  
  sample <- data.frame(cat_pref = y, 
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

``` r
mrp_sim <- simulate_mrp_data(n=1200)
str(mrp_sim)
```

    ## List of 3
    ##  $ sample   :'data.frame':   1200 obs. of  7 variables:
    ##   ..$ cat_pref: num [1:1200] 1 1 0 0 1 1 1 1 1 1 ...
    ##   ..$ male    : num [1:1200] 0 0 1 0 0 0 0 0 0 0 ...
    ##   ..$ age     : num [1:1200] 4 7 7 3 7 5 5 6 6 7 ...
    ##   ..$ eth     : num [1:1200] 3 3 3 1 3 1 3 1 3 3 ...
    ##   ..$ income  : num [1:1200] 1 2 1 2 1 1 3 1 3 1 ...
    ##   ..$ state   : num [1:1200] 19 41 40 19 3 13 13 14 5 36 ...
    ##   ..$ id      : num [1:1200] 1 2 3 4 5 6 7 8 9 10 ...
    ##  $ poststrat:'data.frame':   6300 obs. of  6 variables:
    ##   ..$ male  : num [1:6300] 0 0 0 0 0 0 0 0 0 0 ...
    ##   ..$ eth   : num [1:6300] 1 1 1 1 1 1 1 1 1 1 ...
    ##   ..$ age   : num [1:6300] 1 1 1 1 1 1 1 1 1 1 ...
    ##   ..$ income: num [1:6300] 1 1 1 1 1 1 1 1 1 1 ...
    ##   ..$ state : num [1:6300] 1 2 3 4 5 6 7 8 9 10 ...
    ##   ..$ N     : num [1:6300] 111261 89859 148118 143024 108681 ...
    ##  $ true_popn:'data.frame':   6300 obs. of  6 variables:
    ##   ..$ male    : num [1:6300] 0 0 0 0 0 0 0 0 0 0 ...
    ##   ..$ eth     : num [1:6300] 1 1 1 1 1 1 1 1 1 1 ...
    ##   ..$ age     : num [1:6300] 1 1 1 1 1 1 1 1 1 1 ...
    ##   ..$ income  : num [1:6300] 1 1 1 1 1 1 1 1 1 1 ...
    ##   ..$ state   : num [1:6300] 1 2 3 4 5 6 7 8 9 10 ...
    ##   ..$ cat_pref: num [1:6300] 0.5 0.881 0.622 0.45 0.401 ...
