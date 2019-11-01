from the blog post: https://timmastny.rbind.io/blog/poststratification-with-dplyr/

library(tidyverse)
library(rstan)
library(bayesplot)
library(tidybayes)

### Example 1 with voting data 
vote_yes <- c(rep(0, 25*(1-0.76)), 
              rep(1, 25*0.76),
              rep(0, 50*(1-0.3)),
              rep(1, 50*0.3))
vote_yes

mean(vote_yes)

poll_data <- tibble(
  group = c(rep("A", 25), rep("B", 10)),
  yes = c(rep(1, 19), rep(0, 6), rep(1, 3), rep(0, 7))
)
poll_data %>%
  group_by(group) %>%
  summarise(count = n())

Census <- tibble(
  group = c("A", "B"),
  pop = c(25, 50)
)
Census

mean(poll_data$yes)

group_support <- poll_data %>%
  group_by(group) %>%
  summarise(perc_support = mean(yes))
group_support

overall_support <- group_support %>%
  summarise(total_support = sum(perc_support * Census$pop/sum(Census$pop)))
overall_support

### Example 2 with api survey data 

#Load data 
library(survey)
data(api)

Census <- tibble(
  stype = c("E", "H", "M"),
  pop = c(4421, 755, 1018)
)

d <- apistrat %>% as.tibble()
d %>% 
  group_by(stype) %>% 
  summarise(school_count = n())

d.group.ave <- d %>% 
  group_by(stype) %>%
  summarise(ave_score = mean(api00))
d.group.ave

d.total.ave <- d.group.ave %>%
  summarise(ave_score = sum(ave_score * Census$pop/sum(Census$pop)))
d.total.ave

apipop %>% as.tibble() %>%
  summarise(mean(api00))

