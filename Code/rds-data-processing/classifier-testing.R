library(fastrtext)
library(Matrix)
library(xgboost)
source("R/predict_func.R")
source("R/TweetComplaintPredictor.r")

load("complaint-prediction/models/fullClassifierOne/prcomp4.dat")
fasttextModel <- load_model("complaint-prediction/models/fullClassifierOne/model50-tweet_sample_2M_noRT.dat.bin")
load("complaint-prediction/models/fullClassifierOne/xgb.modelPC4.dat")
TweetComplaintPredictor <- newTweetComplaintPredictor(fasttextModel, train.pr, bst, PC4)

predict(TweetComplaintPredictor, "")