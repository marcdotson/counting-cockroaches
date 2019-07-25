# Simulating Data ---------------------------------------------------------
#simulating data for the model 

library(rethinking)
library(devtools)
library(usethis)

# Simulated data for multiple regression.

# mydata <- sample(1:7, size=1000, replace=TRUE )
#regress <- lm( mydata ~ x1 + x2)
#plot(regress)

#creating variables
#length of tweet
# I set between 10 and 280 because 280 is the max on twitter and 10 seemed like a small enough number 
length_of_tweet <- sample(10:280, size = 1000 , replace = TRUE )

#Have they tweeted about this in the past 

past <- sample(0:1 , size = 1000 , replace = TRUE )


#How many times has this re-tweeted 
re_tweet <- runif(1000, min=0, max=5)

# follower count and re-tweet count may be too correlated to run in the same model 
 
#How many followers they have, I looked it up and the average twitter user has about 700 followers
followers <- rnorm(n = 1000 , mean = 700 , sd = 200 )

# I think we could use rnorm for followers becasue we know the mean (I google it) 
#but for re_tweet it might be better to use a uniform distribution because we don't know anything about how it is distributed 
# I've also aribitraily made our sample size 100 but we can change that if we think something else would be more appropriate 

# Makes sense to me. re_tweet could very well be exponential or normal, but I couldn't give a great reason for either
#I also thought about adding number of negative tweets, or is that the vairable 'past'? M- The variable past was suppose to be if they have tweeted in the past of not

#is there another way to plot this that makes sense?

# Specify parameter values.
beta0 <- 1
beta1 <- 3
beta2 <- 2
beta3 <- 4
beta4 <- 5
error <- rnorm(length(past), 0, 1)

# Generate y.
mydata <- beta0 + beta1 * length + beta2 * past + beta3 * re_tweet + beta4 * followers + error

# Combine into data frame.
d <- data.frame(mydata, length, past, re_tweet, followers)

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
precis(twitter)
plot(twitter)

#trying to run the model without "weights"
plot(mydata)

twitter <- quap(
  alist( 
    mydata ~ dnorm(mu, sigma) , 
    mu <- length + past + re_tweet + followers , 
    sigma ~ dexp(1) 
  ) , data=d) 
precis(twitter)
plot(twitter)



#inserting chapter 5 p146 notation and quap formula

#Ki ∼ Normal(µi, σ)
#µi = α + βNNi + βMMi
#α ∼ Normal(0, 0.2)
#βn ∼ Normal(0, 0.5)
#βm ∼ Normal(0, 0.5)
#σ ∼ Exponential(1)

m5.7 <- quap(
  alist(
    K ~ dnorm( mu , sigma ) ,
    mu <- a + bN*N + bM*M ,
    a ~ dnorm( 0 , 0.2 ) ,
    bN ~ dnorm( 0 , 0.5 ) ,
    bM ~ dnorm( 0 , 0.5 ) ,
    sigma ~ dexp( 1 )
  ) , data=dcc )
precis(m5.7)

# Work on scaling to 1 - 7 
# Making everything have the same impact 
# What the heck this graph is and make it look regression and how to use it for prediction 


