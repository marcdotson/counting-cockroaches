
library(RPostgreSQL)
library(dbplyr)
library(tidyverse)
drv <- dbDriver("PostgreSQL")
con <- dbConnect(drv, dbname = "MarketingAnalyticsClassData", host = "byumarketinganalytics.chaby6kshgs3.us-east-2.rds.amazonaws.com", port = 5555, user = "adrielc", password = "byuanalytics58")
out <- dbSendQuery(con, "SELECT tweet_text, complaint_label FROM tweetcomplaintpreds LEFT JOIN \"Cockroaches\" ON \"Cockroaches\".tweet_id = tweetcomplaintpreds.tweet_id LIMIT 100")
results <- dbFetch(out)
dbDisconnect(con)