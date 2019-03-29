README
================

Dallin Annotations
==================

In this folder, I will keep all of my .rmd documents. Some of them will be created by me, but most will take Adriel's code in other folders and annotate it. I hope that by doing so, I will better understand his code, and be able to contribute more to writing the paper on our reasearch. Additionally, I hope these annotations help future RA's understand the code as well.

The numbering of the files is an attempt to be consistent with the tidyverse style guide section 1.1 (found at <https://style.tidyverse.org/>). As I work more with the code, I will do my best to ensure that the numbering reflects a logical flow for the code.

If the documents only contain a single, massive block of code, this means that I haven't gotten around to annotating that script.

Below, I have a quick and brief documentation of what I understand about each file.

Overview of Files
=================

### 01\_analyze\_complaint\_categories.Rmd

This file analyzes our train/test dataset, documents the label types, visualizes the number of complaints to non-complaints, and visualizes the number of each complaint category. Additionally, it runs a tf\_idf analysis on the catch-all complaint category to see if another category can be identified.

### 02\_create\_clustering\_function.Rmd

### 03\_build\_fasttext\_classification\_model.Rmd

In this file, I create my own fastText supervised classification model to see how well it can classify on its own (i.e. without the help of a PCA and XGBoost model).

Note: Code still has a few places it needs to be tweaked and isn't fully funcitonal yet--I need to write the .txt files and figure out how to make predictions with the trained model.

### 04\_test\_rds\_connection.Rmd

This file contains the credentials necessary to log into the R Data Server. It also has brief code for extracting some of the tweets from the database, and it writes the data to a .txt file.

### 05\_counting\_cockroaches.Rmd

This file takes airline twitter data and creates a few visualizations (i.e. wordclouds) on the tweet text.

### 06\_tweet\_object.Rmd

### 07\_fasttext\_supervised\_classifier.Rmd

This file documents the building of our fastText supervised classifier model.

### 08\_classifier\_construction.Rmd

### 09\_fasttext\_clust\_EDA.Rmd

### 10\_logit\_regression\_classifier.Rmd

### 11\_predict\_func.Rmd

### 12\_tweet\_complaint\_predictor.Rmd

### 13\_batch\_classification.Rmd

### 14\_read\_tweets\_json.Rmd

### 15\_json\_parse\_upload.Rmd

### 16\_classifier\_testing.Rmd

This file references the full model, and makes predictions on single tweets. You can use it to see how the model will react with tweets you find online, as well as tweets you write yourself.
