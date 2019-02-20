library(tidyverse)
library(qtlcharts)
library(caret)
source("R/clus_func.R")
source("R/logit_classifier_func.R")

# read in labeled data ----------------------------------------------------

labeled_data <- read_csv("data/Marketing Research Labeled Tweets_ALL_10-30.csv")


# featurize the data ------------------------------------------------------




# add labels to our embedding data -----------------------------------------

cor_data_50 <- 
  test_out %>% 
  dplyr::select(tweet_text, avg_vec_50) %>% 
  unnest(avg_vec_50) %>% 
  filter(tweet_text != "") %>% 
  bind_cols(labels, .) %>% 
  dplyr::select(-tweet_text1) %>% 
  filter(!is.na(complaint_label)) %>% 
  mutate(label = as.logical(complaint_label))

# feature engineering -----------------------------------------------------

cor_data_50_feat <- 
  cor_data_50 %>% 
  mutate(count.char   = nchar(tweet_text),
         log.mentions = log1p(str_count(tweet_text, "@")), # Count user mentions
         log.links    = log1p(str_count(tweet_text, "http:\\/\\/t.co")), # count links
         count.space  = str_count(tweet_text, " ")) # count exclamation marks

# plot correlations -------------------------------------------------------

iplotCorr(cor_data_50[,-c(1,2)], cor_data_50$label, reorder=TRUE, chartOpts = list(scatcolors=c("lightblue", "lightgreen")))

# Contribution analysis ---------------------------------------------------

beta <- apply(cor_data_50[,-c(1,2)], 2, function(x,y) lm(y~x)$coef[2], cor_data_50$label)
pos <- which(beta > 0)
neg <- which(beta < 0)

iplotCorr(cor_data_50[,-c(1,2)], cor_data_50$label, rows=pos, cols=neg, reorder=TRUE)

# PCA for regression ------------------------------------------------------

cor_data_50.pr <- prcomp(as.matrix(cor_data_50[,3:52]))

# save the prcomp model
cor.data.50.pr <- cor_data_50.pr
save(cor.data.50.pr, file = "models/prcomp/cor.data.50.pr")

screeplot(cor.data.50.pr, type = "line")
# elbow at 6 components. Use 6 components for enriching our model

# Build model -------------------------------------------------------------

PC6_cor_data_50 <- 
  cor_data_50.pr$x %>% 
  as_tibble() %>% 
  select(PC1:PC6)

cor_data_50_feat <- 
  cor_data_50_feat %>%
  bind_cols(PC6_cor_data_50)

beta <- apply(cor_data_50_feat[,-c(1,2)], 2, function(x,y) lm(y~x)$coef[2], cor_data_50$label)
pos <- which(beta > 0)
neg <- which(beta < 0)


cor_data_50_feat %>% 
  select(complaint_label, V1:V50) %>% 
  glm(complaint_label ~ ., data = ., family = "binomial") %>% 
  summary()



iplotCorr(cor_data_50_feat[,-c(1,2)], cor_data_50_feat$label, rows=pos, cols=neg, reorder=TRUE)

# Build model -------------------------------------------------------------

df <- cor_data_50_feat

test_mod <- function(df, testing = FALSE){
  mod <- glm(label ~
               PC2 +
               PC3 +
               PC5 +
               # PC6 + 
               # PC1:PC5 +
               # PC2:PC5 +
               # PC3:PC6 +
               V1 +
               V2 +
               V9 +
               V10 +
               V15 +
               V26 +
               V34 +
               V37 +
               V45 +
               # V46 +
               V48 + 
               V50 +
               # V15:V4 +
               # V34:V31 +
               # V22:V42 +
               # V27:V31 +
               V10:V50 +
               V34:V48 +
               V14:V34 +
               # PC1:V50 +
               # PC1:V35 +
               PC1:V33 +
               PC2:V9 +
               # PC3:V6 +
               PC3:V26 + 
               PC6:V27 +
               # log_mentions + 
               log.links + 
               log.mentions:PC1 + 
               count.char:PC1 + # This one does a lot of the heavy lifting
               count.char:PC4 +
               count.space:PC1,
             data = df, family = "binomial")
  if(testing == TRUE){
    print(summary(mod)) 
  }
  return(mod)
}

accuracy <- list()
for(i in 1:100) { 
  cor_data_50_feat_test_train <- 
    cor_data_50_feat %>% 
    mutate(train1 = rbinom(4958, 1, .75))
  
  mod1_train1 <- 
    cor_data_50_feat_test_train %>% 
    filter(train1 == 1) %>%
    test_mod()
  
  out <- cor_data_50_feat_test_train %>% 
    filter(train1 == 0) %>% 
    mutate(fitted = broom::augment(mod1_train1, newdata = ., type.predict = "response")$.fitted,
           fifty_class = if_else(fitted >= 0.5, TRUE, FALSE))
  
  accuracy[[i]] <- confusionMatrix(factor(as.numeric(out$fifty_class)), factor(as.numeric(out$label)), positive = NULL, dnn = c("Prediction", "Reference"))$overall[1]
}

mean(as.numeric(accuracy))
max(as.numeric(accuracy))
min(as.numeric(accuracy))

# Best so far: 0.8548388


# Save the best performing model ------------------------------------------

logit_mod_0.856_acc <- glm(label ~
      PC2 +
      PC3 +
      PC5 +
      # PC6 + 
      # PC1:PC5 +
      # PC2:PC5 +
      # PC3:PC6 +
      V1 +
      V2 +
      V9 +
      V10 +
      V15 +
      V26 +
      V34 +
      V37 +
      V45 +
      # V46 +
      V48 + 
      V50 +
      # V15:V4 +
      # V34:V31 +
      # V22:V42 +
      # V27:V31 +
      V10:V50 +
      V34:V48 +
      V14:V34 +
      # PC1:V50 +
      # PC1:V35 +
      PC1:V33 +
      PC2:V9 +
      # PC3:V6 +
      PC3:V26 + 
      PC6:V27 +
      # log_mentions + 
      log.links + 
      log.mentions:PC1 +
        count.char:PC1 + # This one does a lot of the heavy lifting
        count.char:PC4 +
        count.space:PC1,
    data = cor_data_50_feat, family = "binomial")

summary(logit_mod_0.856_acc)

save(logit_mod_0.856_acc, file = "models/classifiers/logit_mod_0.856_acc")
