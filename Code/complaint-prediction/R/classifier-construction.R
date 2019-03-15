library(tidyverse)
library(fastrtext)
library(magrittr) # for assignment operator %<>% 
library(plotly)
require(randomForest)
library(Matrix)
source("R/clus_func.R")
source("R/logit_classifier_func.R")


library(xgboost)
library(caret)
library(car)

set.seed(123)


# Fasttext train ----------------------------------------------------------


# train unsupervised fasttext model to learn tweet semantics and word vector representations

# file to read tweets from. Should be a txt file with one tweet per line
file_txt <- "data/tweet_sample_2M_noRT.txt"

# new file to write the trained model to
file_model50  <- "models/fullClassifierOne/model50-tweet_sample_2M_noRT.dat"

# train the unsupervised fasttext model
#fastrtext::execute(commands = c("cbow", "-input", file_txt, "-output", file_model50, "-verbose", 1, "-dim", 50))

# load the model object we just trained
fasttext_model <- load_model(file_model50)


# Fasttext model evaluation -----------------------------------------------

# Examine the nearest neighbors returned by our trained model. 
# First vector element is the word to look up NN, and the second is a string used to filter the results

nn_words  <- list(c("mad", "mad"),
                  c("sad","sad"),
                  c("upset", "upset"),
                  c("late", "ate"),
                  c("delay", "del"),
                  c("cancelled", "cancel"),
                  c("snow", "snow"),
                  c("rain", "rain"),
                  c("weather", "weather"))

map(nn_words, function(x) model_eval_plot_wrap(x[1], fasttext_model, filter_str = x[2], n = 20))

# Predict and average word vectors for tweets -----------------------------

# read in labeled traing data
train <- read_csv("data/Marketing Research Labeled Tweets All 12-05-18.csv")

# Add extracted text features to the dataset ------------------------------

train %<>% 
  mutate(tweetFeatures = map(tweet_text, TweetFeatures))


# create a list column that contains all of features output by the fasttext model
train %<>% 
  mutate(avg_vec_50 = map(tweet_text, VectorizeTweet, model = fasttext_model))


# PCA on fasttext vectors -------------------------------------------------

# select the average vector column and perform a PCA
train.pr <- train %>%
  select(avg_vec_50) %>% 
  unnest(avg_vec_50) %>% 
  as.matrix() %>% 
  prcomp()

# create a dataframe with the pca output to examine the first 3 components

pca_evaluation <- as_data_frame(train.pr$x)
pca_evaluation$label <- factor(train$complaint_label)
pca_evaluation$tweet_text <- train$tweet_text

# evalute the first three components with a 3d plot

plot_ly(pca_evaluation, x = ~PC1, y = ~PC2, z = ~PC3, color = ~label, colors = c('#BF382A', '#0C4B8E'),
        hoverinfo = 'text', text = ~tweet_text) %>% 
  add_markers() %>% 
  layout(scene = list(xaxis = list(title = 'PC1'),
                      yaxis = list(title = 'PC2'),
                      zaxis = list(title = 'PC3')))

# looks good to me. The complaints seperate well.

# make a scree plot of the components
screeplot(train.pr, type = "line", npcs = 20)

# the first component must be the amount of information about a word learned by the fasttext model
# let's go with 15 components with this model.


# Project word vectors onto PCA -------------------------------------------

train %<>% 
  mutate(avg_vec_50_PCA = map(avg_vec_50, VectorizeTweet, model = train.pr, component = PC4))

save(train.pr, file = "models/fullClassifierOne/prcomp4.dat")

save(train, file = "models/fullClassifierOne/train_data.dat")

rm(list = ls())

load("models/fullClassifierOne/train_data.dat")

trainIndex <- createDataPartition(train$complaint_label, p = .8, 
                                  list = FALSE, 
                                  times = 1)


test_data <- train[-trainIndex,]
test_labels <- test_data %>% select(complaint_label) %>% transmute(label = complaint_label)
train_data <- train[trainIndex,]
train_labels <- train_data %>% select(complaint_label) %>% transmute(label = complaint_label)
train_data_vec <- train_data %>%  select(tweetFeatures, avg_vec_50, avg_vec_50_PCA) %>% unnest %>% mutate_all(as.numeric)
test_data_vec <- test_data %>%  select(tweetFeatures, avg_vec_50, avg_vec_50_PCA) %>% unnest %>% mutate_all(as.numeric)

dtrain <- xgb.DMatrix(sparse.model.matrix(~.-1, train_data_vec), label = as.numeric(train_labels$label))
dtest <- xgb.DMatrix(sparse.model.matrix(~.-1, test_data_vec), label = as.numeric(test_labels$label))
dtest1 <- xgb.DMatrix(sparse.model.matrix(~.-1, test_data_vec))
watchlist <- list(train = dtrain, eval = dtest)

logregobj <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  preds <- 1/(1 + exp(-preds))
  grad <- preds - labels
  hess <- preds * (1 - preds)
  return(list(grad = grad, hess = hess))
}

evalerror <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  err <- as.numeric(sum(labels != (preds > 0)))/length(labels)
  return(list(metric = "error", value = err))
}

param <- list(max_depth = 2, eta = 1, silent = 1, nthread = 2, 
              objective = logregobj, eval_metric = evalerror)

bst <- xgboost(data = sparse.model.matrix(~.-1, train_data_vec), label = as.numeric(train_labels$label), max.depth = 4,
               eta = 0.2, nthread = 2, nrounds = 100, objective = logregobj, eval_metric = evalerror)

test_data$prediction <- predict(bst, dtest1)

predict(bst, dtest1) %>% map_chr(function(x) ifelse(x > 0, "complaint", "non-complaint"))
  

test_data %<>%
  mutate(predict_label = case_when(prediction > 0 ~ "Complaint",
                                   TRUE ~ "Non-complaint"),
         complaint_label = case_when(complaint_label == 1 ~ "Complaint",
                                   TRUE ~ "Non-complaint")) %>% 
  mutate(accuracy = predict_label == complaint_label)

confusionMatrix(factor(test_data$complaint_label), factor(test_data$predict_label))

save(bst, file = "models/fullClassifierOne/xgb.modelPC4.dat")
xgb.save(bst, 'models/fullClassifierOne/xgb.modelPC4-xgb_version.dat')
save(test_data, file = "models/fullClassifierOne/test_data.dat")
