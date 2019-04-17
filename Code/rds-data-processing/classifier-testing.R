library(fastrtext)
library(Matrix)
library(xgboost)
source("Code/complaint-prediction/R/predict_func.R")
source("Code/complaint-prediction/R/TweetComplaintPredictor.r")

# load("complaint-prediction/models/fullClassifierOne/prcomp4.dat")
load("Temporary/prcomp4.dat") # Dallin: Location has been edited for where I've kept the model

# fasttextModel <- load_model("complaint-prediction/models/fullClassifierOne/model50-tweet_sample_2M_noRT.dat.bin")
 fasttextModel <- load_model("Temporary/model50-tweet_sample_2M_noRT.dat.bin") # This is where I've kept the saved models


# load("complaint-prediction/models/fullClassifierOne/xgb.modelPC4.dat") # This model was built in Code/complaint-prediction/R/classifier-construction.R
load("Temporary/xgb.modelPC4.dat")
 
TweetComplaintPredictor <- newTweetComplaintPredictor(fasttextModel, train.pr, bst, PC4)

predict(TweetComplaintPredictor, "My mom died this week, and the cartel decapitated my father.")


