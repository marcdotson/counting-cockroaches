# Simulating Data ---------------------------------------------------------
#simulating data for the model 

library(rethinking)
library(devtools)
library(usethis)


#length of tweet
length_of_tweet <- sample(10:280, size = 1000 , replace = TRUE )


#Have they tweeted about this in the past 
past <- sample(0:1 , size = 1000 , replace = TRUE )


#How many times has this re-tweeted 
re_tweet <- runif(1000, min=0, max=5)

 
#How many followers they have
followers <- sample(0:1000, size=1000, replace=TRUE)

#scaling data 
length_of_tweet_scaled <- scale(length_of_tweet)

past_scaled <- scale(past)

re_tweet_scaled <- scale(re_tweet)

followers_scaled <- scale(followers)

View(d_scaled)

# Specify parameter values.
b0 <- 1
bl <- 1
bp <- 1
br <- 1
bf <- 10
error <- rnorm(1000, 0, 1) 

# Generate y.
mydata <- b0 + bl * length_of_tweet + bp * past + br * re_tweet + bf * followers + error

#Generate y_scaled 
mydata_scaled <- b0 + bl *length_of_tweet_scaled + bp * past_scaled + br * re_tweet_scaled + bf * followers_scaled + error

regression <- lm(mydata ~ b0 + bl*length_of_tweet + bp*past + br* re_tweet + bf*followers)

# Combine into data frame.
d <- data.frame(mydata, length_of_tweet, past, re_tweet, followers)

#new scaled data frame
d_scaled <- data.frame(length_of_tweet_scaled,past_scaled,re_tweet_scaled, followers_scaled)

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

#quap scaled 
twitter_scaled <- quap(
  alist( 
    mydata_scaled ~ dnorm(mu, sigma) , 
    mu <- bl*length_of_tweet_scaled + bp*past_scaled + br*re_tweet_scaled + bf*followers_scaled , 
    bl ~ dnorm(0, .25), 
    bp ~ dnorm(0, .25), 
    br ~ dnorm(0, .25), 
    bf ~ dnorm(0, .25), 
    sigma ~ dexp(1) 
  ), 
  data = d_scaled
)


# Evaluating Model Fit ----------------------------------------------------

precis(twitter_scaled)
plot(twitter_scaled)

#trying to run the model without "weights"
plot(mydata_scaled)

impact <- lm(mydata_scaled~followers_scaled, data=d_scaled)

plot(impact)

#Adam 

#McKenna
set.seed(11)
WAIC(twitter_scaled)

set.seed(77) 
compare(twitter_scaled, twitter)




