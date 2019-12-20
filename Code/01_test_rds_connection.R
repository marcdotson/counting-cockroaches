# Overview
# In this document, I will be annotating the code written by Adriel in the script "TestRDSConnection.R."

##### What I understand:
# RDS stands for R Data Server. This script helps us log into the RDS and extract the twitter data. 

# Libraries
library(RPostgreSQL)
library(dbplyr)
library(tidyverse)


# Connect to RDS ----------------------------------------------------------
# This code helps us connect to the server and lost the data tables held inside. Our database is apparently a Postgre database

# Load drivers to communicate with RDS
drv <- dbDriver("PostgreSQL")
con <- dbConnect(drv, dbname = "MarketingAnalyticsClassData",
                 host = "byumarketinganalytics.chaby6kshgs3.us-east-2.rds.amazonaws.com", port = 5555,
                 user = "adrielc", password = "byuanalytics58") # This will prompt user

# List all tables available within the database
dbListTables(con)

### Extract the Data
# With the code chunk above, we were able to see the data tables listed in the server. The dataframe we are interested in is called 
# "Cockroaches." We will find that table and save our connection to that dataframe. This will make it easy for us to wrangle the data on the 
# server before it arrives on our laptop. This will be advantageous because the server will likely have more computing power than our laptops.

# Create tbl object to reference the table serverside when pulling data
Cockroaches <- tbl(con, "Cockroaches")


# Pull Data Sample --------------------------------------------------------


### Pull Data Sample
# Now we will use our connection to wrangle and save the data to our computer.

# Get Tweet text for the first 1,500,000 tweets in the DB. Remove retweets (there's a lot)
sample_tweets <- 
  Cockroaches %>% 
  filter(substr(tweet_text, 1, 4) != "RT @") %>% 
  select(tweet_text) %>% 
  head(1500000) %>% 
  mutate(tweet_text = tolower(tweet_text)) %>%
  collect() # all the processing is done server side before being pulled in by collect()


### Write Tweets to a .txt File
# Now that we have wrangled the data and saved it as an object in our environment, we want to use the write_lines() function to save 
# this data file as a .txt file.

# Once tweets are pulled in, write the lines to a file.
sample_tweets %>%
  .$tweet_text %>%
  write_lines(here::here("Output", "clean_data", "tweet_sample_1hM_noRT.txt"))

# Disconnect
# It is always important do sever your connection to a database after you use it. 

DBI::dbDisconnect(con)

