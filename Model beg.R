#okay here we go try to make a model 
#first I need to figure out how to simulate data 

library(rethinking)
fake.data <- rnorm (1e5 )
simplehist( rnorm , xlab = "fake news")
fake.data <- rnorm(1e4, 4, 2)

set.seed(2)
fake.data <- rnorm(20, 4 , 2)

fake.data

#simulated data for multiple regression 
set.seed(16)
y <- rnorm( n = 100, mean = 0 , sd = 1 )
x1 <- runif( n =100 , min = 1 , max = 2 )
head(x1)
x2 <- runif(n = 100, min = 200 , max =300 )
head(x2)

lm( y ~ x1 + x2)

mydata <- sample(1:7, size=100, replace=TRUE )
regress <- lm( mydata ~ x1 + x2)
plot(regress)

#creating variables 
#length of tweet 
# I set between 10 and 280 because 280 is the max on twitter and 10 seemed like a small enough number 
length <- sample(10:280, size = 100 , replace = TRUE )

#Have they tweeted about this in the past 
past <- sample(1:2 , size = 100 , replace = TRUE )

#How many times has this re-tweeted 
re_tweet <- runif(100, min=0, max=5)

#How many followers they have, I looked it up and the average twitter user has about 700 followers
followers <- rnorm(n = 100 , mean = 700 , sd = 20 )

# I think we could use rnorm for followers becasue we know the man 
#but for re_tweet it might be better to use a normal distribution because we don't know anything about how it is distributed 
# I've also aribitraily made our sample size 100 but we can change that if we think something else would be more appropriate 


fit <- lm( mydata ~ length + past + re_tweet + followers)
summary(fit)
plot(fit)

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


