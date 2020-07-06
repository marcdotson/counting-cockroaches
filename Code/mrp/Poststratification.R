library(tidyverse)

# This is just reference for simple postratification 

# Generate data not in stan. 
mu_1 <- 5
sigma_1 <- 1
mu_2 <- 20 
sigma_2 <- 3

A <- rnorm(100, mu_1, sigma_1)
B <- rnorm(200, mu_2, sigma_2)

# Making the groups. 
percent_A <- mean(A)
percent_B <- mean(B)

# Creating "population" data. 
pop_A <- 200 
pop_B <- 100
total_pop = pop_A + pop_B

# Poststratifying the data.
overall = (percent_A * pop_A/(total_pop)) + (percent_B * pop_B/(total_pop))

# The results.
print(overall)

mean(percent_A, percent_B) # Compare to just averaging over the means of the individual groups. 


