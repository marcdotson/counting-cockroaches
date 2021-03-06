library(tidyverse)
library(fastrtext)
library(stringr)
library(tidytext)


# Testing Models ----------------------------------------------------------
# fasttext.model.50 <- load_model("models/fasttext/model50.dat.bin")
# fasttext.model.50_1 <- load_model("models/model50.dat.bin")
# load("models/prcomp/cor.data.50.pr")
# load("models/classifiers/logit_mod_0.856_acc")

# kTestComplaint <- "@jetblue why are your employees so rude today at dallas-fort worth? tons of attitude on simple questions. #notimpressed"
# kTestNonComplaint <- "@delta I love your service, it's always excellent"

# Featurization -----------------------------------------------------------

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


# Label Prediction -------------------------------------------------------

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



VectorizeTweet <- function(tweet, model, ...) UseMethod("VectorizeTweet", model)

VectorizeTweet.Rcpp_fastrtext <- function(tweet, model, ...) {
  # Computes the word embedding vector for each word in a sentence and then averages them into a single vector
  #
  # Args:
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
  word.vec.tbl <- tbl_df(get_word_vectors(model, tokens))
  # summarise the word vector tibble into one average tibble
  summarise_all(word.vec.tbl, funs(mean), na.rm = TRUE)
}

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
    select(PC1:!!component)
}