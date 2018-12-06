library(RPostgreSQL)
library(dbplyr)

library(R.utils)
library(glue)

library(rjson)
library(RCurl)
library(tidyverse)



# create_df(): ----------------------------------------------------------------
### This function creates one row of a dataframe from a JSON object. This row can be uploaded to a database


create_df <- function(data)
{
  suppressWarnings(
  out <- tibble(
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
    tweet_hastags = null_to_na(glue::collapse(unlist(map(data$entities$hashtags, function(x) x[[1]])), sep = ", ")),
    tweet_urls = null_to_na(glue::collapse(unlist(map(data$entities$urls, function(x) x[[1]])), sep = ", ")),
    tweet_user_mentions = null_to_na(glue::collapse(unlist(map(data$entities$user_mentions, function(x) x[[1]])), sep = ", ")),
    tweet_sentiment = null_to_na(data$sentiment),
    date = null_to_na(data$date$`$date`),
    oid = null_to_na(data$`_id`$`$oid`)
  ))
  return(out)
}


# null_to_na(): -----------------------------------------------------------
### This function converts nulls to NAs for use in building tibbles

is.null_vec <- Vectorize(function(x){is.null(x)}, "x")

null_to_na <- function(x) {
  if(length(x)==0) {
    return(NA)
  }
  return(x)
}

unlist(map(data[[2]]$entities$, function(x) x[[1]]))


# test data import and upload --------------------------------------------------

raw_data <- "data/tweetsFINAL.json"

time <- c()
seqs <- seq(1000, 10000, by = 1000)
for(seq in 1:length(seqs)){
  time[seq] <- system.time({
    data <- fromJSON(sprintf("[%s]", paste(readLines(raw_data, n = seqs[seq]),collapse=",")))
    out <- map_dfr(data, create_df)
  })
}


# data_upload -------------------------------------------------------------

# loads the PostgreSQL driver
drv <- dbDriver("PostgreSQL")
# creates a connection to the postgres database
# note that "con" will be used later in each connection to the database
con <- dbConnect(drv, dbname = "MarketingAnalyticsClassData",
                 host = "byumarketinganalytics.chaby6kshgs3.us-east-2.rds.amazonaws.com", port = 5555,
                 user = "adrielc", password = "byuanalytics58") # This will prompt user

dbListTables(con)


# get number of lines in file ---------------------------------------------

con <- file(raw_data, open = "rb") 
countLines(con, chunkSize = 5e+06)




# upload to database by chunk ---------------------------------------------



seqs <- seq(1730000, 2487767, 10000)
i <- 1
time <- c()
for(seq in seqs) {
  time[i] <- system.time({  
    data <- fromJSON(sprintf("[%s]", paste(read_lines(raw_data, skip = seq, n_max = 10000),collapse=",")))
    upload <- map_dfr(data, create_df) %>%
      distinct(oid, .keep_all = TRUE)
    if(dbExistsTable(con, "Cockroaches")){
      dbWriteTable(con,
                   "Cockroaches",
                   upload,
                   append = T,
                   overwrite = F,
                   row.names = FALSE)
      } else {
        dbWriteTable(con,
                     "Cockroaches",
                     upload,
                     row.names = FALSE)
      }
  })
  i <- i + 1
}

plot(time)

Cockroaches_db <- tbl(con, "Cockroaches")
data_out <- Cockroaches_db %>% summarise(n = n()) %>% collect
data_out
