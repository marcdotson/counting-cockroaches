Postratification Code
================

## Postratification code

We are looking into an MRP model for Counting Cockroaches and we want to
learn what postratification is and if we can do it in our model.

The documents we study are currently together on a google doc

``` r
library(tidyverse)
```

    ## -- Attaching packages --------------------------------------------------------------------------------- tidyverse 1.2.1 --

    ## v ggplot2 3.2.0     v purrr   0.3.2
    ## v tibble  2.1.3     v dplyr   0.8.3
    ## v tidyr   0.8.3     v stringr 1.4.0
    ## v readr   1.3.1     v forcats 0.4.0

    ## -- Conflicts ------------------------------------------------------------------------------------ tidyverse_conflicts() --
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
