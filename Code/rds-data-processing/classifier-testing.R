library(fastrtext)
library(Matrix)
library(xgboost)
source("Code/complaint-prediction/R/predict_func.R")
source("Code/complaint-prediction/R/TweetComplaintPredictor.r")

load("Code/complaint-prediction/models/fullClassifierOne/prcomp4.dat")
fasttextModel <- load_model("Code/complaint-prediction/models/fullClassifierOne/model50-tweet_sample_2M_noRT.dat.bin")
load("Code/complaint-prediction/models/fullClassifierOne/xgb.modelPC4.dat")
TweetComplaintPredictor <- newTweetComplaintPredictor(fasttextModel, train.pr, bst, PC4)


predict(TweetComplaintPredictor, "@delta I hate you sooooo freaking much you made me late to my meeting in Denver")
