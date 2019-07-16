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

#number of tweets about this issue 
count <- sample(0:100, size = 100 , replace = TRUE )

#Have they tweeted about this in the past 
past <- sample(1:2 , size = 100 , replace = TRUE )

#How many times has this re-tweeted 
re_tweet <- rnorm(100, mean = 4 , sd = 2 )

#How many followers they have
followers <- rnorm(n = 100 , mean = 200 , sd = 20 )

fit <- lm( mydata ~ length + count + past + re_tweet + followers)
summary(fit)
plot(fit)

#Hey Adam can you see these changes 
#Here we go again


#Adam back
#let's make the magic happen
#Sorry Marc but github is the worst 

