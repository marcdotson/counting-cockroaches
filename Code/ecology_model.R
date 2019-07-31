# Stan/Rethinking Setup ---------------------------------------------------
# 1. Install Stan, a compiler, and rstan: github.com/stan-dev/rstan/wiki/RStan-Getting-Started.
# 2. Install "experimental" version of the rethinking packages:
#    devtools::install_github("rmcelreath/rethinking", ref = "Experimental")

# Simulating Data ---------------------------------------------------------
# Load packages.
library(rethinking)

# Simulate explanatory variables.
length <- sample(10:280, size = 1000 , replace = TRUE ) # Length of tweet.
past <- sample(0:1 , size = 1000 , replace = TRUE )     # Have they tweeted about this in the past?
retweet <- runif(1000, min=0, max=5)                    # How many times has this been retweeted.
followers <- sample(0:1000, size=1000, replace=TRUE)    # How many followers they have.

# Specify parameter values.
b0 <- 1
bl <- 5
bp <- 10
br <- -3
bf <- 2
error <- rnorm(1000, 0, 1) 

# Generate y.
mydata <- b0 + bl * length + bp * past + br * retweet + bf * followers + error

# Combine into data frame.
data <- data.frame(mydata, length, past, retweet, followers)

# Estimation --------------------------------------------------------------
# Run quap.
out <- quap(
  alist( 
    mydata ~ dnorm(mu, sigma) , 
    mu <- bl * length + bp * past + br * retweet + bf * followers , 
    bl ~ dnorm(0, .25), 
    bp ~ dnorm(0, .25), 
    br ~ dnorm(0, .25), 
    bf ~ dnorm(0, .25), 
    sigma ~ dexp(1) 
  ), 
  data = data
)

# Model output.
out

