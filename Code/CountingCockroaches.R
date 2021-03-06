#################################################
####      1) Data Loading and Preparation     ###
#################################################

# Can we build this without using tm?

library(tidyverse); library(tidytext); library(tm); library(topicmodels)
load("/Users/maddatascientist/DATA SCIENCE/Marketing RA/Twitter Subset2.Rdata") # file location dependent on user
tweets <- twitter_subset

# Format tweets
tweets <- tweets %>% 
  mutate(Tweet_Number = row_number()) %>% 
  select(Tweet_Number, Text = text, Sentiment = sentiment)

# Combine tweets into "mega-tweets" by airline:
airlines <- list(c("americanair", "american", "american air"), "united", "delta", 
                 c("southwestair", "southwest") , "klm", "emirates", "indonesiagaruda", "british_airways", 
                 "usairways", "jetblue", "aircanada", "ryanair", "virginamerica", 
                 c("alaskaairlines", "alaskaair"), "aerlingus", "easyjet","flyfrontier","porterairlines",
                 "asercaairlines","singaporeair", "saudiairlines", "saudiairline", "tamairlines",
                 "cebupacificair","airasia","airfrancefr","koreanairke","deltaassist","aviorairline", 
                 "lufthansa", "airbus")

library(stringr)

docs <- list()
for (i in 1:length(airlines)) {     
  docs[[i]] <- tweets$Text %>%
    str_subset( str_c( "(?i)", str_c( airlines[[i]] ), collapse = "|" ) ) %>% # The "(?i)" is ignoring case
    str_c(collapse = " ")
}

# # #     TAKE OUT ALL AIRLINES     # # #
for (k in 1:length(docs)) {
  for (i in 1:length(airlines)) {
    docs[[k]] <- docs[[k]] %>% 
      str_replace_all( str_c( "@?(?i)", str_c(airlines[[i]]), collapse = "|" ) , "" ) 
  }
}

              # Take Out All "Words" w/ Numbers and All Twitter Handles (optional)

              # docs <- docs %>% 
                # str_replace_all("[\\s]@[^\\s]*", "") %>% # for Twitter handles
                # str_replace_all("[^\\s]*\\d.*?\\s", "") # for all words with numbers


# Get custom stop words

mega_tweets <- docs %>% unlist()

freqs <- list()
for (i in 1:length(mega_tweets)) {    # this loop takes around 4 min
  freq <- mega_tweets[i] %>% 
    VectorSource() %>% 
    Corpus() %>%                
    tm_map(content_transformer(tolower))  %>% 
    tm_map(removePunctuation) %>% 
    tm_map(stripWhitespace) %>% 
    tm_map(removeWords,c(stopwords("english"))) %>% 
    DocumentTermMatrix() %>% 
    as.matrix() %>% 
    colSums() %>% 
    sort(decreasing = TRUE) %>% 
    head(100)
  freqs[[i]] <- freq
}

all_names <- list()
for (l in 1:length(freqs)){
  all_names[[l]] <- names(freqs[[l]])
}

combos <- combn(all_names, 6)

all_intersects <- list()

#----------------------------------------------------------------------
# Parallelizing this task brings processing time from 23 min down to less than 2 min

library(parallel)

intersect_all <- function(a){
  Reduce(intersect, a)
}

no_cores <- detectCores() - 1 # Calculate the number of cores
cl <- makeCluster(no_cores, type="FORK") # Initiate cluster, automatically including environment
all_intersects <- parApply(cl, combos, MARGIN = 2, FUN = intersect_all)
stopCluster(cl)

custom <- Reduce(union, all_intersects)
#----------------------------------------------------------------------

#############################################
####      2) Pre-Model Visualization      ###
#############################################

#----------  Wordcloud by "Mega-tweet" (Airline) ---------- #

library(wordcloud)
library(RColorBrewer)
pal <- brewer.pal(9, "BuGn")[-(1:4)] # colors

freqs <- list()
for (i in 1:length(mega_tweets)) {    # takes about 6-7 min
  freq <- mega_tweets[i] %>% 
    VectorSource() %>% 
    Corpus() %>%                
    tm_map(content_transformer(tolower))  %>% 
    tm_map(removePunctuation) %>% 
    tm_map(stripWhitespace) %>% 
    tm_map(removeWords,c(stopwords("english"), custom)) %>%     # custom stops included here
    DocumentTermMatrix() %>% 
    as.matrix() %>% 
    colSums() %>% 
    sort(decreasing = TRUE) %>% 
    head(100)
  freqs[[i]] <- freq
  wordcloud(words = names(freq), freq = freq, min.freq = min(freq), 
            random.order = FALSE, colors = pal)
}



#############################################
####            3) Model Fitting          ###
#############################################

#Run LDA on DTM

# Get one big dtm from corpora

mega_corpus <- mega_tweets %>% 
  VectorSource() %>% 
  Corpus()

mega_corpus <- mega_corpus %>%                # Takes about 6 min on 31 docs
  tm_map(content_transformer(tolower))  %>% 
  tm_map(removePunctuation) %>% 
  tm_map(stripWhitespace) %>% 
  tm_map(removeWords,c(stopwords("english"), custom)) # can use custom list too

mega_dtm <- DocumentTermMatrix(mega_corpus)

#----------------------------------------------------------------------
# Process run sequentially

# Fit LDA for different numbers of topics
num_topics <- 2:20
num_terms <- 20
topic_results <- list()
terms <- list()
for ( k in num_topics ){                # takes 43 min on 18 docs
  topic_results[[k-1]] <- mega_dtm %>% 
    LDA(method = "Gibbs", k = k, control = list(seed = 42))
  
  terms[[k-1]] <- topic_results[[k-1]] %>% 
    terms(k = num_terms) 
}
#----------------------------------------------------------------------

#----------------------------------------------------------------------
# Parallelizing this task has processing time of 40 min ... no improvement on sequential

library(parallel)
library(doParallel)

fit_LDAs <- function(dtm, k){
  dtm %>% 
    LDA(method = "Gibbs", k = k, control = list(seed = 42))
}

num_topics <- 2:20

registerDoParallel(cores = 3)
topic_results <- foreach(k = num_topics) %do% fit_LDAs(mega_dtm,k)

#----------------------------------------------------------------------

betas <- list()
gammas <- list()
for ( n in 1:length(topic_results) ) {
  betas[[n]] <- topic_results[[n]] %>% tidy(matrix = "beta")
  gammas[[n]] <- topic_results[[n]] %>% tidy(matrix = "gamma") %>% 
    group_by(document) %>% 
    top_n(1, gamma) %>% 
    ungroup()
}

for ( j in 1:length(gammas) ) {
  cat(j+1, "Topic(s)")
  cat("\n") 
  gammas[[j]] %>% arrange(as.numeric(document)) %>% print()
  cat("\n") 
}

for ( j in 1:length(betas) ) {
  cat(j+1, "Topic(s)")
  cat("\n") 
  betas[[j]] %>% print()
  cat("\n") 
}

# Optimize Number of Topics (from https://cran.r-project.org/web/packages/ldatuning/vignettes/topics.html)

library(ldatuning)
result <- FindTopicsNumber(
  mega_dtm,
  topics = seq(from = 2, to = 20, by = 1),
  metrics = c("Griffiths2004", "CaoJuan2009", "Arun2010", "Deveaud2014"),
  method = "Gibbs",
  control = list(seed = 77),
  mc.cores = 2L,
  verbose = TRUE
)

FindTopicsNumber_plot(result) # 11 topics then ?


#################################################
####      4) Post-Model Visualization    ###
#################################################

# Top Terms by Topic

library(ggplot2)
library(dplyr)

new_betas <- list()
for ( m in 1:length(betas) ){
  
  new_betas[[m]] <- betas[[m]] %>% 
    filter(beta > 0.001) %>% 
    arrange(as.numeric(topic))
  
}

top_terms <- list()
for (k in 1:length(betas)) {
  
  top_terms[[k]] <- new_betas[[k]] %>%
    group_by(topic) %>%
    top_n(10, beta) %>%
    ungroup() %>%
    arrange(topic, -beta)
  
}

for ( t in 1:length(top_terms) ) {
  print(  
    top_terms[[t]] %>%
      mutate(term = reorder(term, beta)) %>%
      ggplot(aes(term, beta, fill = factor(topic))) +
      geom_col(show.legend = FALSE) +
      facet_wrap( ~topic, scales = "free") +
      coord_flip()
  )
}

