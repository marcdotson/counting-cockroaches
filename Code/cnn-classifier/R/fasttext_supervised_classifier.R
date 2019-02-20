library(fastrtext)
library(tidyverse)

# Prepare data ------------------------------------------------------------

# Create file location where the model will be written to
file_model <- "models/class_mod_1.dat"
train_data <- "data/train_data_1.txt"
test_data <- "data/test_data_1.txt"

# read in the data
tweet_data <- read_csv("data/Marketing Research Labeled Tweets_ - tweet_sample_5k_FULL.csv") %>% 
  filter(nchar(tweet_text) > 5)

# format the data to fasttext input requirements
test_train_data <- 
  tweet_data %>% 
  mutate(label = case_when(label == 0 ~ "non-complaint", # each line (tweet) is preceded by a label to be predicted
                              label == 1 ~ "complaint"),
            input_lines = paste(label, tweet_text)) %>% 
  group_by(label) %>% # get a representative sample of both complaints and non complaints
  mutate(test_train = c("test", "train")[rbinom(n(), 1, 0.8) + 1]) %>% # assign test train splits
  ungroup()

# write train data
train_data_lines <- 
  test_train_data %>% 
  filter(test_train == "train") %>% 
  pull(input_lines) %>% 
  paste0("__label__", .)

write_lines(train_data_lines, train_data)

# write test data
test_data_lines <- 
  test_train_data %>% 
  filter(test_train == "test") %>% 
  pull(input_lines) %>% 
  paste0("__label__", .)

write_lines(test_data_lines, test_data)

# create a character vector containing the tweets to test without their labels
test_labels_without_prefix <- 
  test_train_data %>% 
  filter(test_train == "test") %>% 
  pull(label)
  

# learn model
execute(commands = c("supervised", "-input", train_data, "-output", file_model, "-dim", 20, "-lr", 1, "-epoch", 20, "-wordNgrams", 2, "-verbose", 1))

# load model
model <- load_model(file_model)

# prediction are returned as a list with words and probabilities
predictions <- predict(model, sentences = test_data_lines, simplify = TRUE, unlock_empty_predictions = TRUE)
head(predictions, 5)

length(predictions)
length(test_labels_without_prefix)
length(test_data_lines)

# Compute accuracy
mean(names(unlist(predictions)) == test_labels_without_prefix, na.rm = TRUE)

# because there is only one category by observation, hamming loss will be the same
get_hamming_loss(as.list(test_labels_without_prefix), predictions)

# test predictions
predictions <- predict(model, sentences = test_to_write)
print(head(predictions, 5))

# you can get flat list of results when you are retrieving only one label per observation
print(head(predict(model, sentences = test_to_write, simplify = TRUE)))
