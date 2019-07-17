#simulating data for the model 

#All this is just getting all the right packages installed
install.packages("usethis")

remove.packages("rstan")
if (file.exists(".RData")) file.remove(".RData")

install.packages("rstan", repos = "https://cloud.r-project.org/", dependencies = TRUE)
install.packages("processx")

install.packages(c("coda","mvtnorm","devtools"))
library(devtools)
devtools::install_github("rmcelreath/rethinking",ref="Experimental")
1

library("rstan")

install.packages(c("mvtnorm","loo","coda"), repos="https://cloud.r-project.org/",dependencies=TRUE)
options(repos=c(getOption('repos'), rethinking='http://xcelab.net/R'))
install.packages('rethinking',type='source')


install.packages("rethinking")


library(rethinking)
library(devtools)
library(usethis)

#that should do it

library(rethinking)

#simulated data for multiple regression 

mydata <- sample(1:7, size=1000, replace=TRUE )
#regress <- lm( mydata ~ x1 + x2)
#plot(regress)

#creating variables 
#length of tweet 
# I set between 10 and 280 because 280 is the max on twitter and 10 seemed like a small enough number 
length <- sample(10:280, size = 1000 , replace = TRUE )

#Have they tweeted about this in the past 
past <- sample(1:2 , size = 1000 , replace = TRUE )

#How many times has this re-tweeted 
re_tweet <- runif(1000, min=0, max=5)

#How many followers they have, I looked it up and the average twitter user has about 700 followers
followers <- rnorm(n = 1000 , mean = 700 , sd = 200 )

# I think we could use rnorm for followers becasue we know the mean (I google it) 
#but for re_tweet it might be better to use a uniform distribution because we don't know anything about how it is distributed 
# I've also aribitraily made our sample size 100 but we can change that if we think something else would be more appropriate 
Adam
# Makes sense to me. re_tweet could very well be exponential or normal, but I couldn't give a great reason for either
#I also thought about adding number of negative tweets, or is that the vairable 'past'?
#is there another way to plot this that makes sense?

#combine into data frame 
d <- data.frame(length, past, re_tweet, followers)
d

twitter <- quap(
  alist( 
    mydata ~ dnorm(mu, sigma) , 
    mu <- bl*length + bp*past + br*re_tweet + bf*followers , 
    bl ~ dnorm(0, .25), 
    bp ~ dnorm(0, .25), 
    br ~ dnorm(0, .25), 
    bf ~ dnorm(0, .25), 
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


