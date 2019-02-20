library(googlesheets)
library(plotly)
library(broom)
library(caret)
library(cluster)
library(qtlcharts)
library(RPostgreSQL)
library(dbplyr)
source("R/clus_func.R")
source("R/rquery_cormat.R")


# connect to db -----------------------------------------------------------


drv <- dbDriver("PostgreSQL")
con <- dbConnect(drv, dbname = "MarketingAnalyticsClassData",
                 host = "byumarketinganalytics.chaby6kshgs3.us-east-2.rds.amazonaws.com", port = 5555,
                 user = "adrielc", password = "byuanalytics58") # This will prompt user
dbListTables(con)



# pull data sample --------------------------------------------------------


Cockroaches <- tbl(con, "Cockroaches")

sample_tweets <- 
  Cockroaches %>% 
  select(tweet_text) %>% 
  mutate(tweet_text = tolower(tweet_text)) %>%
  filter(substr(tweet_text, 1, 4) != "rt @") %>% 
  head(900000) %>% 
  collect()


sample_tweets %>%
  .$tweet_text %>% 
  write_lines("data/tweet_sample_1mil.txt")


# Remove retweets
sample_tweets <- 
  Cockroaches %>% 
  filter(substr(tweet_text, 1, 4) != "RT @") %>% 
  select(tweet_text) %>% 
  head(1500000) %>% 
  mutate(tweet_text = tolower(tweet_text)) %>%
  collect()

sample_tweets %>%
  .$tweet_text %>%
  write_lines("data/tweet_sample_1hM_noRT.txt")
rm(sample_tweets)



# Remove retweets
sample_tweets <- 
  Cockroaches %>% 
  filter(substr(tweet_text, 1, 4) != "RT @") %>% 
  select(tweet_text) %>% 
  head(2000000) %>% 
  mutate(tweet_text = tolower(tweet_text)) %>%
  collect()

sample_tweets %>%
  .$tweet_text %>%
  write_lines("data/tweet_sample_2M_noRT.txt")
rm(sample_tweets)






# create_clusters ---------------------------------------------------------


file_txt <- "data/tweet_sample_1hM_noRT.txt"

file_model200 <- "data/models/model200.dat"
file_model150 <- "data/models/model150.dat"
file_model100 <- "data/models/model100.dat"
file_model50  <- "data/models/model50.dat"
file_model25  <- "data/models/mode25.dat"

# execute(commands = c("cbow", "-input", file_txt, "-output", file_model200, "-verbose", 1, "-dim", 200))
model200 <- load_model(file_model200)

# execute(commands = c("cbow", "-input", file_txt, "-output", file_model150, "-verbose", 1, "-dim", 150))
model150 <- load_model(file_model150)

# execute(commands = c("cbow", "-input", file_txt, "-output", file_model100, "-verbose", 1, "-dim", 100))
model100 <- load_model(file_model100)

# execute(commands = c("cbow", "-input", file_txt, "-output", file_model50, "-verbose", 1, "-dim", 50))
model50 <- load_model(file_model50)

# execute(commands = c("cbow", "-input", file_txt, "-output", file_model25, "-verbose", 1, "-dim", 25))
model25 <- load_model(file_model25)



 
# test word extraction
dict <- get_dictionary(model50)

# test word vectors for specific words
print(get_word_vectors(model50, c("delta", "americanair")))

# Nearest-neighbor queries for words
model_eval("mad", model150, model100, model50, model25, filter_str = "mad") %>% 
  model_eval_plot(20, "mad", "mad")


# Nearest-neighbor queries for words --------------------------------------

nn_words  <- 
  list(c("mad", "mad"),
       c("sad","sad"),
       c("upset", "upset"),
       c("late", "ate"),
       c("delay", "del"),
       c("cancelled", "cancel"),
       c("snow", "snow"),
       c("rain", "rain"),
       c("weather", "weather"))

map(nn_words, function(x) model_eval_plot_wrap(x[1], model200, model150, model100, model50, model25, filter_str = x[2], n = 20))

model_eval_plot_wrap("baggage", model200, model150, model100, model25, filter_str = "bag")
model_eval_plot_wrap("delay", model200, model150, model100, model50, model25, filter_str = "del")
model_eval_plot_wrap("delay", model200, model150, model100, model50, model25)
model_eval_plot_wrap("tarmac", model200, model150, model100, model50, model25, filter_str = "tarmac")

# create list columns with average vectors for tweets ---------------------

sample_tweets <- 
  read_lines("data/tweet_sample_1hM_noRT.txt") %>% 
  tibble(tweet_text)

system.time({
  vec_out <- sample_tweets %>%
    sample_n(5000) %>% 
    mutate(avg_vec_200 = map(tweet_text, avg_word_vec, model = model200),
           avg_vec_150 = map(tweet_text, avg_word_vec, model = model150),
           avg_vec_100 = map(tweet_text, avg_word_vec, model = model100),
           avg_vec_50  = map(tweet_text, avg_word_vec, model = model50),
           avg_vec_25  = map(tweet_text, avg_word_vec, model = model25))
})


test_out <- test_out %>% mutate(avg_vec_25 = map(tweet_text, avg_word_vec, model = model25))

# plot WSS for kmeans solutions -------------------------------------------


set.seed(123)

# Compute and plot wss for k = 1 to k = 15
k.values <- 1:7

# extract wss for 2-7 clusters for each model type
wss_values150 <- map_dbl(k.values, wss_func(2, vec_out))
wss_values100 <- map_dbl(k.values, wss_func(3, vec_out))
wss_values50 <- map_dbl(k.values, wss_func(4, vec_out))
par(mfrow=c(2,2))
plot(k.values, wss_values150,
     type="b", pch = 19, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares",
     main = "Embedding 150")
plot(k.values, wss_values100,
     type="b", pch = 19, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares",
     main = "Embedding 100")
plot(k.values, wss_values50,
     type="b", pch = 19, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares",
     main = "Embedding 50")



# assign clusters to tweets -----------------------------------------------

vec_labeled <- kmeans_assign(test_out, 6:11, 2:6)

# vec_labeled %>% 
#   select(tweet_text, k6_avg_vec_200:k11_avg_vec_25) %>% 
#   write_csv("data/labeled_data_3.csv")

vec_labeled <- read_csv("data/labeled_data_2.csv")

# Randomly sample from the clusters to determine which contain complaints --------



# This labeled data is from "data/labeled_data_2.csv"
# HOW THESE WERE LABELED
# Every cluster gets 1 of 2 labels: def, maybe. No label applied if confident a cluster shouldn't belong
# Each sample of 20 is evaluated with the following heuristics:
# def: If more than 70% of the sample is complaints (percentages are not actually calculated for each cluster, just assumed)
# maybe: if more than 1 complaint is found and is less than 70%

# Here are some examples of complaint types:
## Lost luggage: people are without there stuff and want compensation
## Flight delay: Want some recompense for hotels
## What to do about a missed flight
## Ticket issues
## On the Tarmac waiting
## Poor customer service
## No response from the airline
## Failure in contacting through phone
## Extra/hidden fees

# 3 main reasons that people complain
# Want a response for information
# Want a response for compensation
# Social retaliation

# Interesting note: a lot of complaints contain positive words but are written with sarcasm

# 6 clusters
sample_num_levels(vec_labeled, 7,  20) # 1, 6 - maybe: 2, 4
sample_num_levels(vec_labeled, 8,  20) # 1, 6 - maybe: 4
sample_num_levels(vec_labeled, 9,  20) # 2, 5 - maybe: 4
sample_num_levels(vec_labeled, 10, 20) # 4, 5 - maybe: 3
sample_num_levels(vec_labeled, 11, 20) # 3, 6

# 7 clusters
sample_num_levels(vec_labeled, 12, 20) # 3    - maybe: 2
sample_num_levels(vec_labeled, 13, 20) # 3, 6 - maybe: 7
sample_num_levels(vec_labeled, 14, 20) # 1    - maybe: 6, 7
sample_num_levels(vec_labeled, 15, 20) # 2, 6 - maybe: 7
sample_num_levels(vec_labeled, 16, 20) # 5    - maybe: 3

# 8 clusters
sample_num_levels(vec_labeled, 17, 20) # 1, 4    - maybe: 5, 8
sample_num_levels(vec_labeled, 18, 20) # 2, 3    - maybe: 4, 6
sample_num_levels(vec_labeled, 19, 20) # 1, 3, 7 - maybe: 2, 6
sample_num_levels(vec_labeled, 20, 20) # 3, 4    - mqybe: 6
sample_num_levels(vec_labeled, 21, 20) # 3       - maybe: 1, 8

# 9 clusters
sample_num_levels(vec_labeled, 22, 20) # def: 6, 7, 8 - maybe: 1, 8
sample_num_levels(vec_labeled, 23, 20) # def: 6, 7, 9 - maybe: 1
sample_num_levels(vec_labeled, 24, 20) # def: 4, 5, 7 - maybe: 3
sample_num_levels(vec_labeled, 25, 20) # def: 5, 7, 8 - maybe:
sample_num_levels(vec_labeled, 26, 20) # def: 1, 7, 8 - maybe:

# 10 clusters
sample_num_levels(vec_labeled, 27, 20) # def: 1, 3, 4     - maybe: 6
sample_num_levels(vec_labeled, 28, 20) # def: 3, 4, 7     - maybe: 9 
sample_num_levels(vec_labeled, 29, 20) # def: 2, 4, 5, 7  - maybe:
sample_num_levels(vec_labeled, 30, 20) # def: 1, 6, 8     - maybe: 4, 10
sample_num_levels(vec_labeled, 31, 20) # def: 4, 10       - maybe: 3

# 11 clusters
sample_num_levels(vec_labeled, 32, 20) # def: 2, 3, 6     - maybe: 1
sample_num_levels(vec_labeled, 33, 20) # def: 2, 7, 8, 9  - maybe:
sample_num_levels(vec_labeled, 34, 20) # def: 1, 2, 9     - maybe: 4, 5
sample_num_levels(vec_labeled, 35, 20) # def: 6, 7, 9     - maybe:
sample_num_levels(vec_labeled, 36, 20) # def: 5, 7, 8     - maybe:




# Implement this set logic into getting a decently labeled dataset ------------------

num_def_overlap <- 
  vec_labeled %>% 
  transmute(tweet_text = tweet_text,
            summy = apply(., 1, function(x) {sum(
  # 6 clusters
  x$k6_avg_vec_200 %in% c(1, 6), 
  x$k6_avg_vec_150 %in% c(1, 6), 
  x$k6_avg_vec_100 %in% c(2, 5),
  x$k6_avg_vec_50  %in% c(4, 5), 
  x$k6_avg_vec_25  %in% c(3, 6), 
  
  # 7 clusters
  x$k7_avg_vec_200 %in% c(3), 
  x$k7_avg_vec_150 %in% c(3, 6), 
  x$k7_avg_vec_100 %in% c(1),
  x$k7_avg_vec_50  %in% c(2, 6), 
  x$k7_avg_vec_25  %in% c(5), 
  
  # 8 clusters
  x$k8_avg_vec_200 %in% c(1, 4), 
  x$k8_avg_vec_150 %in% c(2, 3), 
  x$k8_avg_vec_100 %in% c(1, 3, 7),
  x$k8_avg_vec_50  %in% c(3, 4), 
  x$k8_avg_vec_25  %in% c(3), 
  
  # 9 clusters
  x$k9_avg_vec_200 %in% c(6, 7, 8), 
  x$k9_avg_vec_150 %in% c(6, 7, 9), 
  x$k9_avg_vec_100 %in% c(4, 5, 7),
  x$k9_avg_vec_50  %in% c(5, 7, 8), 
  x$k9_avg_vec_25  %in% c(1, 7, 8), 
  
  # 10 clusters
  x$k10_avg_vec_200 %in% c(1, 3, 4), 
  x$k10_avg_vec_150 %in% c(3, 4, 7), 
  x$k10_avg_vec_100 %in% c(2, 4, 5, 7),
  x$k10_avg_vec_50  %in% c(1, 6, 8), 
  x$k10_avg_vec_25  %in% c(4, 10), 
  
  # 11 clusters
  x$k11_avg_vec_200 %in% c(2, 3, 6), 
  x$k11_avg_vec_150 %in% c(2, 7, 8, 9), 
  x$k11_avg_vec_100 %in% c(1, 2, 9),
  x$k11_avg_vec_50  %in% c(6, 7, 9), 
  x$k11_avg_vec_25  %in% c(5, 7, 8)
  )}))

# PLot the number of tweets within a group 

hist(num_def_overlap$summy, xlim = c(1, 25), density = sapply(num_def_overlap$tweet_text, str_count, pattern = "\\S+"), breaks = 20)

ggplot(num_def_overlap, aes(x = summy)) +
  geom_density()

length(num_def_overlap[which(num_def_overlap$summy < 2),]$tweet_text) # 1037
sample_n(num_def_overlap[which(num_def_overlap$summy < 2),], 20) # 

length(num_def_overlap[which(num_def_overlap$summy > 22),]$tweet_text) # 987
sample_n(num_def_overlap[which(num_def_overlap$summy > 22),], 20)








vec_labeled <- 
  vec_labeled %>% 
  mutate(
    # label a cluster group as probably containing most all complaints
    def_20 = ifelse(
      apply(., 1, function(x) {sum(
        # 6 clusters
        x$k6_avg_vec_200 %in% c(1, 6), 
        x$k6_avg_vec_150 %in% c(1, 6), 
        x$k6_avg_vec_100 %in% c(2, 5),
        x$k6_avg_vec_50  %in% c(4, 5), 
        x$k6_avg_vec_25  %in% c(3, 6), 
        
        # 7 clusters
        x$k7_avg_vec_200 %in% c(3), 
        x$k7_avg_vec_150 %in% c(3, 6), 
        x$k7_avg_vec_100 %in% c(1),
        x$k7_avg_vec_50  %in% c(2, 6), 
        x$k7_avg_vec_25  %in% c(5), 
        
        # 8 clusters
        x$k8_avg_vec_200 %in% c(1, 4), 
        x$k8_avg_vec_150 %in% c(2, 3), 
        x$k8_avg_vec_100 %in% c(1, 3, 7),
        x$k8_avg_vec_50  %in% c(3, 4), 
        x$k8_avg_vec_25  %in% c(3), 
        
        # 9 clusters
        x$k9_avg_vec_200 %in% c(6, 7, 8), 
        x$k9_avg_vec_150 %in% c(6, 7, 9), 
        x$k9_avg_vec_100 %in% c(4, 5, 7),
        x$k9_avg_vec_50  %in% c(5, 7, 8), 
        x$k9_avg_vec_25  %in% c(1, 7, 8), 
        
        # 10 clusters
        x$k10_avg_vec_200 %in% c(1, 3, 4), 
        x$k10_avg_vec_150 %in% c(3, 4, 7), 
        x$k10_avg_vec_100 %in% c(2, 4, 5, 7),
        x$k10_avg_vec_50  %in% c(1, 6, 8), 
        x$k10_avg_vec_25  %in% c(4, 10), 
        
        # 11 clusters
        x$k11_avg_vec_200 %in% c(2, 3, 6), 
        x$k11_avg_vec_150 %in% c(2, 7, 8, 9), 
        x$k11_avg_vec_100 %in% c(1, 2, 9),
        x$k11_avg_vec_50  %in% c(6, 7, 9), 
        x$k11_avg_vec_25  %in% c(5, 7, 8)
      )}) > 20, TRUE, FALSE))
      


vec_labeled <- 
  vec_labeled %>% 
  mutate(
    # label a group as definitely not containing complaints
    def_not_0_1 = ifelse(
      apply(., 1, function(x) {sum(
        # 6 clusters
        x$k6_avg_vec_200 %in% c(1, 6), 
        x$k6_avg_vec_150 %in% c(1, 6), 
        x$k6_avg_vec_100 %in% c(2, 5),
        x$k6_avg_vec_50  %in% c(4, 5), 
        x$k6_avg_vec_25  %in% c(3, 6), 
        
        # 7 clusters
        x$k7_avg_vec_200 %in% c(3), 
        x$k7_avg_vec_150 %in% c(3, 6), 
        x$k7_avg_vec_100 %in% c(1),
        x$k7_avg_vec_50  %in% c(2, 6), 
        x$k7_avg_vec_25  %in% c(5), 
        
        # 8 clusters
        x$k8_avg_vec_200 %in% c(1, 4), 
        x$k8_avg_vec_150 %in% c(2, 3), 
        x$k8_avg_vec_100 %in% c(1, 3, 7),
        x$k8_avg_vec_50  %in% c(3, 4), 
        x$k8_avg_vec_25  %in% c(3), 
        
        # 9 clusters
        x$k9_avg_vec_200 %in% c(6, 7, 8), 
        x$k9_avg_vec_150 %in% c(6, 7, 9), 
        x$k9_avg_vec_100 %in% c(4, 5, 7),
        x$k9_avg_vec_50  %in% c(5, 7, 8), 
        x$k9_avg_vec_25  %in% c(1, 7, 8), 
        
        # 10 clusters
        x$k10_avg_vec_200 %in% c(1, 3, 4), 
        x$k10_avg_vec_150 %in% c(3, 4, 7), 
        x$k10_avg_vec_100 %in% c(2, 4, 5, 7),
        x$k10_avg_vec_50  %in% c(1, 6, 8), 
        x$k10_avg_vec_25  %in% c(4, 10), 
        
        # 11 clusters
        x$k11_avg_vec_200 %in% c(2, 3, 6), 
        x$k11_avg_vec_150 %in% c(2, 7, 8, 9), 
        x$k11_avg_vec_100 %in% c(1, 2, 9),
        x$k11_avg_vec_50  %in% c(6, 7, 9), 
        x$k11_avg_vec_25  %in% c(5, 7, 8)
      )}) < 2, TRUE, FALSE))



vec_labeled <- 
  vec_labeled %>% 
  mutate(
    # label a group as definitely not containing complaints
    def_22 = ifelse(
      apply(., 1, function(x) {sum(
        # 6 clusters
        x$k6_avg_vec_200 %in% c(1, 6), 
        x$k6_avg_vec_150 %in% c(1, 6), 
        x$k6_avg_vec_100 %in% c(2, 5),
        x$k6_avg_vec_50  %in% c(4, 5), 
        x$k6_avg_vec_25  %in% c(3, 6), 
        
        # 7 clusters
        x$k7_avg_vec_200 %in% c(3), 
        x$k7_avg_vec_150 %in% c(3, 6), 
        x$k7_avg_vec_100 %in% c(1),
        x$k7_avg_vec_50  %in% c(2, 6), 
        x$k7_avg_vec_25  %in% c(5), 
        
        # 8 clusters
        x$k8_avg_vec_200 %in% c(1, 4), 
        x$k8_avg_vec_150 %in% c(2, 3), 
        x$k8_avg_vec_100 %in% c(1, 3, 7),
        x$k8_avg_vec_50  %in% c(3, 4), 
        x$k8_avg_vec_25  %in% c(3), 
        
        # 9 clusters
        x$k9_avg_vec_200 %in% c(6, 7, 8), 
        x$k9_avg_vec_150 %in% c(6, 7, 9), 
        x$k9_avg_vec_100 %in% c(4, 5, 7),
        x$k9_avg_vec_50  %in% c(5, 7, 8), 
        x$k9_avg_vec_25  %in% c(1, 7, 8), 
        
        # 10 clusters
        x$k10_avg_vec_200 %in% c(1, 3, 4), 
        x$k10_avg_vec_150 %in% c(3, 4, 7), 
        x$k10_avg_vec_100 %in% c(2, 4, 5, 7),
        x$k10_avg_vec_50  %in% c(1, 6, 8), 
        x$k10_avg_vec_25  %in% c(4, 10), 
        
        # 11 clusters
        x$k11_avg_vec_200 %in% c(2, 3, 6), 
        x$k11_avg_vec_150 %in% c(2, 7, 8, 9), 
        x$k11_avg_vec_100 %in% c(1, 2, 9),
        x$k11_avg_vec_50  %in% c(6, 7, 9), 
        x$k11_avg_vec_25  %in% c(5, 7, 8)
      )}) > 22, TRUE, FALSE))



# write_csv(select(vec_labeled, tweet_text, def_20, def_not_0_1, def_22), "data/labeled_clusters_train_1.csv")

load("data/vectorized_tweet_sample_5k.dat")
labeled_clusters_train_1 <- read_csv("data/labeled_clusters_train_1.csv")
labels <- read_csv("data/Marketing Research Labeled Tweets_All.csv")

vec_labeled <- cbind(labeled_clusters_train_1, labels)[,-6]

vec_labeled %>% 
  filter(!is.na(label)) %>% 
  dplyr::group_by(def_20) %>% 
  dplyr::summarise(accuracy = sum(label)/sum(vec_labeled$label, na.rm = T))


vec_labeled %>% 
  filter(!is.na(label)) %>% 
  dplyr::group_by(def_22) %>% 
  dplyr::summarise(accuracy = sum(label)/sum(vec_labeled$label, na.rm = T))

vec_labeled %>% 
  filter(!is.na(label)) %>% 
  dplyr::group_by(def_not_0_1) %>% 
  dplyr::summarise(accuracy = sum(label)/sum(vec_labeled$label, na.rm = T))

vec_labeled %>% 
  filter(label == 1,
         def_not_0_1 == TRUE) %>%
  select(tweet_text)


vec_labeled[which(grepl("se niega a dejar", vec_labeled$tweet_text)),]$label

vec_labeled[which(grepl("se niega a dejar", vec_labeled$tweet_text)),]$label

labels[which(grepl("se niega a dejar", labels$tweet_text)),]$label

# plot correlations and proportions ---------------------------------------

labels <- read_csv("data/Marketing Research Labeled Tweets_All.csv")
load("data/vectorized_tweet_sample_5k.dat")

# correlation avg_vec_25 --------------------------------------------------

cor_data_25 <- 
  test_out %>% 
  select(tweet_text, avg_vec_25) %>% 
  unnest(avg_vec_25) %>% 
  filter(tweet_text != "") %>% 
  bind_cols(labels, .) %>% 
  select(-tweet_text1) %>% 
  filter(!is.na(label)) %>% 
  mutate(lebel = as.logical(label))

iplotCorr(cor_data_25[,-c(1,2)], cor_data_25$label, reorder=TRUE, chartOpts = list(scatcolors=c("lightblue", "lightgreen")))


# Contribution analysis ---------------------------------------------------
beta <- apply(cor_data_25[,-c(1,2)], 2, function(y,x) lm(y~x)$coef[2], cor_data_25$label)
pos <- which(beta > 0)
neg <- which(beta < 0)


iplotCorr(cor_data_25[,-c(1,2)], cor_data_25$label, rows=factor(pos), cols=factor(neg), reorder=TRUE)


# Correlation avg_vec_50 --------------------------------------------------

cor_data_50 <- 
  test_out %>% 
  dplyr::select(tweet_text, avg_vec_50) %>% 
  unnest(avg_vec_50) %>% 
  filter(tweet_text != "") %>% 
  bind_cols(labels, .) %>% 
  dplyr::select(-tweet_text1) %>% 
  filter(!is.na(label)) %>% 
  mutate(label = as.logical(label))

iplotCorr(cor_data_50[,-c(1,2)], cor_data_50$label, reorder=TRUE, chartOpts = list(scatcolors=c("lightblue", "lightgreen")))

lm(label ~ V8, data = cor_data_50)$coef[2]

# Contribution analysis ---------------------------------------------------

beta <- apply(cor_data_50[,-c(1,2)], 2, function(x,y) lm(y~x)$coef[2], cor_data_50$label)
pos <- which(beta > 0)
neg <- which(beta < 0)

iplotCorr(cor_data_50[,-c(1,2)], cor_data_50$label, rows=pos, cols=neg, reorder=TRUE)

# Looks like
mod1 <- glm(label ~ V2 + V10 + V2:V10, data = cor_data_50, family = "binomial")
mod2 <- glm(label ~ V2 + V10, data = cor_data_50, family = "binomial")
mod3 <- glm(label ~ V2*V10 + V11:V10 + V34:V48 + V11*V4 + V32, data = cor_data_50, family = "binomial")

mod3 <- glm(label ~ V12 + V34*V48 + V45 + V35 + V2, data = cor_data_50, family = "binomial")

summary(mod1)
summary(mod2)
summary(mod3)

## Test Cases
# tweet_text <- "@mcdo##naldsarabia @emirates @thedubaimall its really crazy, i'm argantina football fan and now i'm the argantini burger fan also üòç"
# tweet_text <- "vuelo d@e a@yer@@@@@ con @vueling??? vistas de!!!!! #acoru√±a #galicia http://t.co/60rqbl2nc8 http://t.co/60rqbl2nc8 http://t.co/60rqbl2nc8 http://t.co/60rqbl2nc8 http://t.co/60rqbl2nc8"

# featurize that shiz
cor_data_50_feat <- 
  cor_data_50 %>% 
  mutate(hashtags = str_count(tweet_text, "#"), # Count hashtags
         log_mentions = log1p(str_count(tweet_text, "@")), # Count user mentions
         log_links    = log1p(str_count(tweet_text, "http:\\/\\/t.co")), # count links
         question     = str_count(tweet_text, "\\?"), # count question marks 
         exclaim      = str_count(tweet_text, "!")) # count exclamation marks

mod4 <- glm(label ~ V2*V10 + V11:V10 + V34:V48 + V11*V4 + V32 + hashtags + log_mentions + log_links, data = cor_data_50_feat, family = "binomial")

mod5 <- glm(label ~ V12 + V34*V48 + V45 + V35 + V2 + log_mentions + log_links, data = cor_data_50_feat, family = "binomial")
summary(mod5)


accuracy <- list()
for(i in 1:100) {
  cor_data_50_feat_test_train <- 
    cor_data_50_feat %>% 
    mutate(train1 = rbinom(4959, 1, .7))
  
  mod5_train1 <- 
    cor_data_50_feat_test_train %>% 
    filter(train1 == 1) %>%
    glm(label ~ V12 + V34*V48 + V45 + V35 + V2 + log_mentions + log_links, data = ., family = "binomial")
  
  out <- cor_data_50_feat_test_train %>% 
    filter(train1 == 0) %>% 
    mutate(fitted = broom::augment(mod5_train1, newdata = ., type.predict = "response")$.fitted,
           fifty_class = if_else(fitted >= 0.5, TRUE, FALSE))
  
  accuracy[[i]] <- confusionMatrix(factor(as.numeric(out$fifty_class)), factor(as.numeric(out$label)), positive = NULL, dnn = c("Prediction", "Reference"))$overall[1]
}

mean(as.numeric(accuracy))
max(as.numeric(accuracy))
min(as.numeric(accuracy))




# a -----------------------------------------------------------------------

unnest(test_out, avg_vec_25)

cor_data_50_feat_test_train <- 
  cor_data_50_feat %>% 
  mutate(train1 = rbinom(4959, 1, .7))

which(cor_data_50_feat_test_train$train1 == 1)

which(cor_data_50_feat_test_train$train1 == 1)

unnest()

cor_data_50_feat_test_train %>% 
  filter(train1 == 1)


mod5_train1 <- 
  cor_data_50_feat_test_train %>% 
  filter(train1 == 1) %>%
  glm(label ~ V12 + V34*V48 + V45 + V35 + V2 + log_mentions + log_links, data = ., family = "binomial")

out <- cor_data_50_feat_test_train %>% 
  filter(train1 == 0) %>% 
  mutate(fitted = broom::augment(mod5_train1, newdata = ., type.predict = "response")$.fitted, 
         fifty_class = if_else(fitted >= 0.5, TRUE, FALSE))



confusionMatrix(factor(as.numeric(out$fifty_class)), factor(as.numeric(out$label)), positive = NULL, dnn = c("Prediction", "Reference"))$overall[1]


out %>%
  filter(fifty_class == FALSE, label == TRUE) %>% 
  select(tweet_text) %>% 
  sample_n(20)


clus_4_50_mod5_train1 <- kmeans(filter(out, fifty_class == TRUE)[,3:52], centers = 4, iter.max = 30, nstart = 15)

clustered_complaints_mod50_mod5_train1 <- 
  out %>%
  filter(fifty_class == TRUE) %>% 
  mutate(clus_4 = kmeans(.[,3:52], centers = 4, iter.max = 30, nstart = 15)$cluster,
         clus_3 = kmeans(.[,3:52], centers = 3, iter.max = 30, nstart = 15)$cluster,
         clus_2 = kmeans(.[,3:52], centers = 2, iter.max = 30, nstart = 15)$cluster)

clustered_complaints_mod50_mod5_train1 %>% 
  ggplot(aes(x = clus_4, fill = factor(clus_4))) +
  geom_bar()

clustered_complaints_mod50_mod5_train1 %>% 
  ggplot(aes(x = clus_3, fill = factor(clus_4))) +
  geom_bar()

clustered_complaints_mod50_mod5_train1 %>% 
  ggplot(aes(x = clus_2, fill = factor(clus_4))) +
  geom_bar()

clustered_complaints_mod50_mod5_train1 %>% 
  sample_num_levels(sample_col_num = 62, 20)

clustered_complaints <- 
  cor_data_50_feat %>% 
  filter(label == TRUE) %>% 
  transmute(tweet_text = tweet_text,
            clus_2 = kmeans(.[,3:52], centers = 2, iter.max = 30, nstart = 15)$cluster,
            clus_3 = kmeans(.[,3:52], centers = 3, iter.max = 30, nstart = 15)$cluster,
            clus_4 = kmeans(.[,3:52], centers = 4, iter.max = 30, nstart = 15)$cluster,
            clus_5 = kmeans(.[,3:52], centers = 5, iter.max = 30, nstart = 15)$cluster,
            clus_6 = kmeans(.[,3:52], centers = 6, iter.max = 30, nstart = 15)$cluster)

clustered_complaints %>% 
  sample_num_levels(value_col_num = 1, sample_col_num = 6, n = 20)


# Try to ascertain what complaint types we have ---------------------------
## We want to first create clusters that are defined by their distance from certain words
baggage_complaint_words <- c("bag", "luggage", "baggage", "suitcase", "lost", "clothes", "checked", "bags", "losing", "lose")
baggage_complaint_vec <- avg_word_vec(baggage_complaint_words, model50)
delay_complaint_words <- c("delay", "hour", "mechanical", "tarmac", "time", "minute", "cancelled", "announce", "layover")
delay_complaint_vec <- avg_word_vec(delay_complaint_words, model50)
service_complaint_words <- c("service", "customer", "relations", "dirty", "food", "minute", "rep", "phone", "attendant", "rude", "serve", "unprofessional")
service_complain_vec <- avg_word_vec(service_complaint_words, model50)
website_booking_complaint_words <- c("website", "internet", "online", "app", "book", "booking", "reservation", "ticket", "expensive")
website_booking_complaint_vec <- avg_word_vec(website_booking_complaint_words, model50)

labels <- read_csv("data/Marketing Research Labeled Tweets_All.csv")
load("data/vectorized_tweet_sample_5k.dat")

cor_data_50 <- 
  test_out %>% 
  dplyr::select(tweet_text, avg_vec_50) %>% 
  unnest(avg_vec_50) %>% 
  filter(tweet_text != "") %>% 
  bind_cols(labels, .) %>% 
  dplyr::select(-tweet_text1) %>% 
  filter(!is.na(label)) %>% 
  mutate(label = as.logical(label))

cor_data_50 <- 
  cor_data_50 %>% 
  mutate(bag_dist = map_dbl(tweet_text, function(text, vec, model) sqrt(sum((avg_word_vec(text, model)-vec)^2)), baggage_complaint_vec, model50),
         del_dist = map_dbl(tweet_text, function(text, vec, model) sqrt(sum((avg_word_vec(text, model)-vec)^2)), delay_complaint_vec, model50),
         srv_dist = map_dbl(tweet_text, function(text, vec, model) sqrt(sum((avg_word_vec(text, model)-vec)^2)), service_complain_vec, model50),
         web_dist = map_dbl(tweet_text, function(text, vec, model) sqrt(sum((avg_word_vec(text, model)-vec)^2)), website_booking_complaint_vec, model50),
         bag_dist_scale = scale(bag_dist),
         del_dist_scale = scale(del_dist),
         srv_dist_scale = scale(srv_dist),
         web_dist_scale = scale(web_dist))

cor_data_50_clus <- 
  cor_data_50 %>% 
  filter(label == 1) %>% 
  mutate(clus_2 = kmeans(.[,53:56], centers = 2, iter.max = 30, nstart = 15)$cluster,
         clus_3 = kmeans(.[,53:56], centers = 3, iter.max = 30, nstart = 15)$cluster,
         clus_4 = kmeans(.[,53:56], centers = 4, iter.max = 30, nstart = 15)$cluster,
         clus_5 = kmeans(.[,53:56], centers = 5, iter.max = 30, nstart = 15)$cluster,
         clus_6 = kmeans(.[,53:56], centers = 6, iter.max = 30, nstart = 15)$cluster,
         luggage = kmeans(.[,57], centers = 10, iter.max = 30, nstart = 15)$cluster,
         delays  = kmeans(.[,58], centers = 10, iter.max = 30, nstart = 15)$cluster,
         service = kmeans(.[,59], centers = 10, iter.max = 30, nstart = 15)$cluster,
         website = kmeans(.[,60], centers = 10, iter.max = 30, nstart = 15)$cluster)

cor_data_50_clus %>% 
  ggplot(aes(x = luggage, fill = factor(clus_2))) +
  geom_bar()

cor_data_50_clus %>%
  filter(label == 1) %>% 
  sample_num_levels(sample_col_num = 66)



cor_data_50_clus %>% 
  filter(label == 1) %>% 
  ggplot(aes(x = clus_2)) +
  geom_bar()


data(trees)
trees

# Test out principal components -------------------------------------------

cor_data_50.pr <- prcomp(as.matrix(cor_data_50[,3:52]))

PC_cor_data_50 <- as_data_frame(cor_data_50.pr$x)
PC_cor_data_50$label <- factor(cor_data_50$label)
PC_cor_data_50$tweet_text <- cor_data_50$tweet_text

plot_ly(PC_cor_data_50, x = ~PC1, y = ~PC2, z = ~PC3, color = ~label, colors = c('#BF382A', '#0C4B8E'),
        hoverinfo = 'text', text = ~tweet_text) %>% 
  add_markers() %>% 
  layout(scene = list(xaxis = list(title = 'PC1'),
                      yaxis = list(title = 'PC2'),
                      zaxis = list(title = 'PC3')))




scatterplot3d(cor_data_50.pr$x[,1:3], color = factor(cor_data_50$label))


plot(cor_data_50.pr$x[,1], col = factor(cor_data_50$label))


# create a test train pipeline


na_filtered_vec_lab_2 %>% 
  iplotCorr(unnest(.[,5]), .$label, reorder=TRUE)
