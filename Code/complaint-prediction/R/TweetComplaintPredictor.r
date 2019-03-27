# TweetComplaintPredictor class constructor ----------------------------------------
# Class constructor 
newTweetComplaintPredictor <- function(fasttextModel, pcaModel, xgbModel, pcaModelComponentCutoff) {
  component <- enquo(pcaModelComponentCutoff)
  
  if(class(fasttextModel)[1] != "Rcpp_fastrtext"){
    stop("fasttextModel is not a member of class Rcpp_fastrtext")
  }
  if(class(pcaModel) != "prcomp"){
    stop("pcaModel is not a member of class Rcpp_fastrtext")
  }
  if(class(xgbModel) != "xgb.Booster"){
    stop("xgbModel is not a member of class Rcpp_fastrtext")
  }
  
  newTweetComplaintPredictor <- list(fasttextModel = fasttextModel,
                                     pcaModel = pcaModel,
                                     xgbModel = xgbModel,
                                     pcaModelComponentCutoff = component)
  
  class(newTweetComplaintPredictor) <- "TweetComplaintPredictor"
  
  return(newTweetComplaintPredictor)
}

# Class checker
is.TweetComplaintPredictor <- function(x) inherits(x, "TweetComplaintPredictor")


