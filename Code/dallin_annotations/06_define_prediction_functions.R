
# Overview ----------------------------------------------------------------
# This script defines three different functions that were frequently referenced when building our classification model. When this script is 
# referenced in other parts of code, it's treated as a package to import these nine functions.


# Libraries ---------------------------------------------------------------

library(tidyverse)
library(fastrtext)
library(stringr)
library(tidytext)


# Testing Models ----------------------------------------------------------
# I imagine that Adriel left this section to quickly test his functions as he built them.

# fasttext.model.50 <- load_model("models/fasttext/model50.dat.bin")
# fasttext.model.50_1 <- load_model("models/model50.dat.bin")
# load("models/prcomp/cor.data.50.pr")
# load("models/classifiers/logit_mod_0.856_acc")

# kTestComplaint <- "@jetblue why are your employees so rude today at dallas-fort worth? tons of attitude on simple questions. #notimpressed"
# kTestNonComplaint <- "@delta I love your service, it's always excellent"

# TweetFeatures Function -----------------------------------------------------------



TweetFeatures <- function(tweet) {
  # Creates features from the tweet which are used for prediction
  # 
  # Args:
  #   tweet: a single tweet as a string
  # 
  # Returns:
  #   a single-row tibble with the following columns: log.links, log.mentions, count.char, count.space
  tibble(tweet.text = tweet) %>%
    mutate(count.char   = nchar(tweet.text),
           log.mentions = log1p(str_count(tweet.text, "@")), # Count user mentions
           log.links    = log1p(str_count(tweet.text, "http:\\/\\/t.co")), # count links
           count.space  = str_count(tweet.text, " ")) %>% # count exclamation marks
    select(-tweet.text)
}


# Label Prediction Function -------------------------------------------------------

predict.TweetComplaintPredictor <- function(object, newdata, complaintCutoff = 0) {
  # Given a character vector containing tweets to predict for, this function returns predicted labels for the given tweet.
  #
  # Args:
  #   object: TweetComplaintPredictor model
  #   newdata: A character vector containing tweets  
  #   complaintCutoff: a numeric value specifying the cutoff value for the xgb prediction output 
  #       when considering what value to use to consider a tweet a complaint.
  #
  # Returns:
  #   a string label. Either "Complaint" or "Non-Complaint"
  #
  
  newdata %>% 
    tibble(tweetText = .) %>% 
    mutate(tweetFeatures = map(tweetText, TweetFeatures)) %>% 
    mutate(fasttextVec = map(tweetText, VectorizeTweet, object$fasttextModel)) %>% 
    mutate(pcaVec = map(fasttextVec, VectorizeTweet, object$pcaModel, object$pcaModelComponentCutoff)) %>% 
    unnest %>% 
    select(-tweetText) %>% 
    sparse.model.matrix(~.-1, .) %>% 
    xgb.DMatrix %>% 
    predict(object$xgbModel, .) %>% 
    map_chr(function(x) ifelse(x > complaintCutoff, "Complaint", "Non-Complaint"))
}


# Vectorize Tweet ---------------------------------------------------------
# What It Does: This function is very similar to the avg_word_vec() function in 05_define_clustering_functions(), except that it has two
# classes: one for handling fastText models, and another for handling PCA models. The fastText class takes the word vectors of the full tweet 
# and returns a single, averaged vector. The PCA class returns the first n principal components where n is the number of components defined 
# by the user. In our case, we used a scree plot to see how many principal components we should look at. 

VectorizeTweet <- function(tweet, model, ...) UseMethod("VectorizeTweet", model)

VectorizeTweet.Rcpp_fastrtext <- function(tweet, model, ...) {
  # Computes the word embedding vector for each word in a sentence and then averages them into a single vector
  #
  # Arguments:
  #   tweet: a single tweet as a string
  #   model: a trained fasttext model used to compute the word embedding vectors
  # 
  # Returns:
  #   a single-row tibble with each embedding vector element as a column. The number of rows is
  #   defined by the dim parameter that was used to train the input fasttext model
  #
  # tokenize the text, return a list of tokens
  tokens <- strsplit(tweet, " ")[[1]]
  # apply get_word_vectors to each token in the vector, return tibble of word vecs, one row for each word
  # word.vec.tbl <- tbl_df(get_word_vectors(model, tokens)) Dallin: This code doesn't work for me, so I replaced it with the line below.
  word.vec.tbl <- as_tibble(get_word_vectors(model, tokens))
  # summarise the word vector tibble into one average tibble
  summarise_all(word.vec.tbl, funs(mean), na.rm = TRUE)
}
# Dallin: This VectorizeTweet function is not working for me. The error I get comes from when I try to use the tbl_df() function on line 81
# Dallin Update: I think I've fixed it? It now runs as expected on line 80 of classifier-construction.R

VectorizeTweet.prcomp <- function(tweet, model, component, ...) {
  # Computes the principal components from the embedding vectors and returns the first 6
  #
  # Args:
  #   df: a tibble of word embedding vectors
  #   prcomp.mod: a trained prcomp model used to compute principal components of the word embedding vectors
  # 
  # Returns:
  #   a single-row tibble with N principal component variables
  #
  # error handling: embedding vector must be of length 50
  
  predict(model, newdata = tweet) %>% 
    as_tibble() %>% 
    select(PC1:component) # select(PC1:!!component) # Dallin: the code that didn't work has been replaced with code I think will work.
    
  
  # Dallin: I don't think this function is working right. I keep getting an error when I run this function around line 
  # 135 of the classifier-construction.R script
  # I don't know exactly where the predict() function comes from, but I imagine it predicts based on the prc model.
  
}


# Logregobj Function ------------------------------------------------------
# Dallin: Logistic Regression on Object (I think?)

logregobj <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  preds <- 1/(1 + exp(-preds))
  grad <- preds - labels
  hess <- preds * (1 - preds)
  return(list(grad = grad, hess = hess))
}


# Evalerror Function ------------------------------------------------------


evalerror <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  err <- as.numeric(sum(labels != (preds > 0)))/length(labels)
  return(list(metric = "error", value = err))
}