library(RPostgreSQL)
library(dbplyr)


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


# Pull Data Sample --------------------------------------------------------

# Get Tweet text for the first 1500000 tweets in the DB. Remove retweets (there's a lot)
sample_tweets <- 
  Cockroaches %>% 
  filter(substr(tweet_text, 1, 4) != "RT @") %>% 
  select(tweet_text) %>% 
  head(1500000) %>% 
  mutate(tweet_text = tolower(tweet_text)) %>%
  collect() # all the processing is done server side before being pulled in by collect()

# Once tweets are pulled in, write the lines to a file.
sample_tweets %>%
  .$tweet_text %>% 
  write_lines("data/tweet_sample_1hM_noRT.txt")

