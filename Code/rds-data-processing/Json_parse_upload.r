library(RPostgreSQL)
library(dbplyr)
library(rjson)
library(RCurl)
library(tidyverse)



# create_df(): ----------------------------------------------------------------
### This function creates one row of a dataframe from a JSON object. This row can be uploaded to a database


create_df <- function(data)
{
  tibble(
    tweet_created_at = null_to_na(data$created_at),
    tweet_id = null_to_na(data$id_str),
    tweet_text = null_to_na(data$text),
    in_reply_to_status_id_str = null_to_na(data$in_reply_to_status_id_str),
    in_reply_to_user_id_str = null_to_na(data$in_reply_to_user_id_str),
    in_reply_to_screen_name = null_to_na(data$in_reply_to_screen_name),
    usr_name = null_to_na(data$user$name),
    usr_screen_name = null_to_na(data$user$screen_name),
    usr_url = null_to_na(data$user$url),
    usr_description = null_to_na(data$user$description),
    usr_protected = null_to_na(data$user$protected),
    usr_followers_count = null_to_na(data$user$followers_count),
    usr_friends_count = null_to_na(data$user$friends_count),
    usr_listed_count = null_to_na(data$user$listed_count),
    usr_created_at = null_to_na(data$user$created_at),
    usr_statuses_count = null_to_na(data$user$statuses_count),
    usr_verified = null_to_na(data$user$verified),
    coordinates = null_to_na(data$coordinates),
    tweet_retweet_count_at_pull_time = null_to_na(data$retweet_count),
    tweet_favorite_count_at_pull_time = null_to_na(data$favorite_count),
    tweet_hastags = null_to_na(data$entities$hashtags),
    tweet_symbols = null_to_na(data$entities$symbols),
    tweet_urls = null_to_na(data$entities$urls),
    tweet_user_mentions = null_to_na(data$entities$user_mentions),
    tweet_sentiment = null_to_na(data$sentiment),
    date = null_to_na(data$date$`$date`),
    oid = null_to_na(data$`_id`$`$oid`)
  )
}

null_to_na <- function(x) ifelse(is.null(x), NA, x)

# data import and upload --------------------------------------------------



raw_data <- "data/tweetsFINAL.json"

data <- fromJSON(sprintf("[%s]", paste(readLines(raw_data, n = 1000),collapse=",")))

out <- map_dfr(data, create_df)











# data_upload -------------------------------------------------------------

# loads the PostgreSQL driver
drv <- dbDriver("PostgreSQL")
# creates a connection to the postgres database
# note that "con" will be used later in each connection to the database
con <- dbConnect(drv, dbname = "MarketingAnalyticsClassData",
                 host = "byumarketinganalytics.chaby6kshgs3.us-east-2.rds.amazonaws.com", port = 5555,
                 user = "adrielc", password = "byuanalytics58") # This will prompt user

dataUpload <- read_csv("../DataAcquisitionProject/Nielsen Retail Fact Data.csv")


dbListTables(con)
