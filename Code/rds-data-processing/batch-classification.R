library(RPostgreSQL)
library(dbplyr)
library(tidyverse)
source("complaint-prediction/R/TweetComplaintPredictor.r")
source("complaint-prediction/R/predict_func.R")

# Connect to RDS ----------------------------------------------------------

# Load drivers to communicate with RDS
drv <- dbDriver("PostgreSQL")
con <- dbConnect(drv, dbname = "MarketingAnalyticsClassData",
                 host = "byumarketinganalytics.chaby6kshgs3.us-east-2.rds.amazonaws.com", port = 5555,
                 user = "adrielc", password = "byuanalytics58") # This will prompt user

# List all tables available within the database
dbListTables(con)

# Create tbl object to reference the table serverside when pulling data
Cockroaches <- tbl(con, "Cockroaches")
test <- tbl(con, "NielsenRetail")

dbSendQuery(con, "ALTER TABLE \"Cockroaches\" ADD COLUMN complaint_label VARCHAR;")

res <- dbSendQuery(con, "insert into \"NielsenRetail\" (complaint_label) values ('test', 1, 1, 1);")


Cockroaches %>% 
  select(tweet_id, tweet_text) %>% 
  collect()

sample <- all_tweets %>% head(100)


  
load("complaint-prediction/models/fullClassifierOne/prcomp4.dat")
fasttextModel <- load_model("complaint-prediction/models/fullClassifierOne/model50-tweet_sample_2M_noRT.dat.bin")
load("complaint-prediction/models/fullClassifierOne/xgb.modelPC4.dat")
TweetComplaintPredictor <- newTweetComplaintPredictor(fasttextModel, train.pr, bst, PC4)

sample %<>% 
  mutate(prediction = predict(TweetComplaintPredictor, tweet_text))




system.time({sample %<>% 
    mutate(prediction = predict(TweetComplaintPredictor, tweet_text))
})