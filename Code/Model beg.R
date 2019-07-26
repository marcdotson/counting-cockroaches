# Simulating Data ---------------------------------------------------------
# Load packages.
library(rethinking)
library(devtools)
library(usethis)

# Simulate explanatory variables.
length_of_tweet <- sample(10:280, size = 1000 , replace = TRUE ) # Length of tweet.
past <- sample(0:1 , size = 1000 , replace = TRUE )              # Have they tweeted about this in the past.
re_tweet <- runif(1000, min=0, max=5)                            # How many times has this re-tweeted.
# followers <- sample(0:1000, size=1000, replace=TRUE)             # How many followers they have.
followers <- sample(0:1000, size=1000, replace=TRUE)             # How many followers they have.

# Specify parameter values.
b0 <- 1
bl <- 5
bp <- 10
br <- -3
bf <- 2
error <- rnorm(1000, 0, 1) 

# Generate y.
mydata <- b0 + bl * length_of_tweet + bp * past + br * re_tweet + bf * followers + error

# Combine into data frame.
d <- data.frame(mydata, length_of_tweet, past, re_tweet, followers)

# Estimation --------------------------------------------------------------
# Run quap.
twitter <- quap(
  alist( 
    mydata ~ dnorm(mu, sigma) , 
    mu <- bl*length_of_tweet + bp*past + br*re_tweet + bf*followers , 
    bl ~ dnorm(0, .25), 
    bp ~ dnorm(0, .25), 
    br ~ dnorm(0, .25), 
    bf ~ dnorm(0, .25), 
    sigma ~ dexp(1) 
  ), 
  data = d
)

# Evaluating model fit.
precis(twitter_scaled)
plot(twitter_scaled)

#trying to run the model without "weights"
plot(mydata_scaled)

impact <- lm(mydata_scaled~followers_scaled, data=d_scaled)

plot(impact)

#Adam 

#McKenna

WAIC(twitter_scaled)

compare(twitter, twitter_scaled)


