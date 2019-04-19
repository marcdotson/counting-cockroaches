library(fastrtext)
library(Matrix)
library(xgboost)
source("Code/complaint-prediction/R/predict_func.R")
source("Code/complaint-prediction/R/TweetComplaintPredictor.r")

load("Temporary/prcomp4.dat") # Dallin: Location has been edited for where I've kept the model

fasttextModel <- load_model("Temporary/model50-tweet_sample_2M_noRT.dat.bin") # This is where I've kept the saved models

load("Temporary/xgb.modelPC4.dat")
 
TweetComplaintPredictor <- newTweetComplaintPredictor(fasttextModel, train.pr, bst, PC4)

predict(TweetComplaintPredictor, "My mom died this week, and the cartel decapitated my father.")
