
# Overview ----------------------------------------------------------------
# This script defines nine different functions that were frequently referenced when building our classification model. When this script is 
# referenced in other parts of code, it's treated as a package to import these nine functions.


# Libraries ---------------------------------------------------------------

library(fastrtext)
library(tidyverse)
library(cluster)    # clustering algorithms
library(factoextra) # clustering visualization
library(dendextend) # for comparing two dendrograms
library(parallel)
library(pbapply)
library(stringr)


# Avg_word_veg Function ---------------------------------------------------
# What it does:
    # An unsupervised fasttext model creates word embeddings for each word in its training dataset. These word embeddings can be thought of 
    # as mathematical definitions of words. The unsupervised model we trained used a word vector of 50 variables. This function, 
    # avg_word_vec(), takes the word vectors for every word in a tweet, and averages each of the 50 variables. These averages create a single 
    # vector that can be thought of as a mathematical summary of the tweet. This single vector becomes a useful input for our PCA model.

# Arguments: 
    # Text:  A tweet
    # Model: A trained unsupervised FastText model.

avg_word_vec <- function(text, model) {
  # tokenize the text, return a list of tokens
  vec_of_tokens <- strsplit(text, " ")[[1]]
  # apply get_word_vectors to each token in the vector, return matrix of word vecs
  matrix_of_vecs <- get_word_vectors(model, vec_of_tokens)
  # convert matrix to a tibble for easy reduction
  tbl_of_vecs <- tbl_df(matrix_of_vecs) 
  # summarise the word vector tibble into one average tibble
  summarise_all(tbl_of_vecs, funs(mean), na.rm = TRUE)
}


# Nn_filter Function ------------------------------------------------------


nn_filter <- function(model, token, filter_str, n = 1000) {
  out <- get_nn(model, token, n)
  if(!is.na(filter_str)) {
    tibble(word = names(out), value = out) %>% 
      filter(!grepl(paste0(".*", filter_str, ".*"), word, ignore.case = T))
  } else {
    tibble(word = names(out), value = out)
  }
}


# Model_eval Function -----------------------------------------------------


model_eval <- function(token, ..., filter_str = NA) {
  models <- list(...)
  names(models) <- GetDotObjectNames(...)
  cl <- detectCores()
  pblapply(models, nn_filter, token, filter_str, cl = cl)
}


# GetDotObjectNames Function ----------------------------------------------


GetDotObjectNames <- function(...) {
  strsplit(
    gsub("[\\(\\)]", 
         "", 
         regmatches(
           deparse(
             substitute(
               list(...))), 
           gregexpr("\\(.*?\\)", 
                    deparse(
                      substitute(
                        list(...)))))[[1]]), 
    split = ", ")[[1]]
}



# Model_eval_plot Function ------------------------------------------------


model_eval_plot <- function(eval_list, n = 20, token, filter_str) {
  out <- lapply(names(eval_list), function(model) {
    eval_list[[model]] %>% 
      mutate(model = model)
  }) %>% 
    bind_rows() %>% 
    group_by(model) %>% 
    top_n(n, value) %>% 
    ungroup() %>% 
    arrange(model, value) %>% 
    mutate(order = row_number())
  
  out %>% 
    ggplot(aes(order, value, fill = model)) +
    geom_bar(stat = "identity", show.legend = FALSE) +
    facet_wrap(~model, scales = "free") +
    labs(y = "Euclidean Dist",
         x = NULL) +
    ggtitle(paste("Top", n, "Closest Words"), subtitle = paste0("In Euclidean Distance from '", token, "'", ifelse(missing(filter_str) || is.na(filter_str), "", paste0(" with '",filter_str,"' filtered out")))) +
    scale_x_continuous(
      breaks = out$order,
      labels = out$word,
      expand = c(0,0)
    ) +
    coord_flip()
}


# Model_eval_plot_wrap Function -------------------------------------------


model_eval_plot_wrap <- function(token, ..., filter_str = NA, n = 20) {
  model_eval(token, ..., filter_str = filter_str) %>% 
    model_eval_plot(n, token, filter_str)
}


# Wss_func Function -------------------------------------------------------


# function factory to compute total within-cluster sum of square, given a list column number
wss_func <- function(data, col_num) {
  function(k) {
    kmeans(unnest(data[,col_num]), k, nstart = 10 )$tot.withinss
  }
}


# Kmeans_assign Function ---------------------------------------------------


kmeans_assign <- function(data, kseq, colseq, nstart = 15, iter.max = 30) {
  df_length <- nrow(data)
  data <- filter(data, tweet_text != "")
  len_diff <- df_length - nrow(data)
  if(len_diff > 0) {
    message(paste0("Filtered ", len_diff, " num rows of blank strings"))
  }
  data_colnames <- colnames(data)
  for(i in kseq) {
    for(j in colseq) {
      varname <- paste0("k", i, "_", data_colnames[j])
      data <- mutate(data, !!varname := kmeans(unnest(data[,j]), i, nstart = nstart, iter.max = iter.max)$cluster)
    }
  }
  return(data)
}


# Sample_num_levels Function ----------------------------------------------


sample_num_levels <- function(df, sample_col_num, n = 20) {
  for(i in sample_col_num) {
    col_levels <- max(df[,i])
    for(j in 1:col_levels) {
      out <- 
        df %>% 
        .[which(.[,i] == j),] %>% 
        dplyr::select(tweet_text) %>% 
        sample_n(n, replace = T)
      cat(paste("\n\nSampling", n, "observations from cluster", j, "in column ", colnames(df)[i],"\n\n"))
      print(out)
    }
  }
}


