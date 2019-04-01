

library(here)
library(tidyverse)
library(quanteda)
library(keras)
library(yardstick)


# ------- setup ----------
get_done_raw <- read_csv(here("data", "get_it_done_2018_requests_datasd.csv"))

# remove rows with NA
get_done <- get_done_raw %>%
  filter(!is.na(public_description), !is.na(service_name))

get_done <- get_done %>%
  select(service_request_id, service_name, public_description)

# what categories of issues are there?
get_done %>% 
  count(service_name, sort = TRUE) %>% 
  View()

# let's just look at categories with more than 5000 complaints
# we will also exclude 'Other' because that doesn't tell us what the issue is
keepers <- c("72 Hour Violation", "Graffiti Removal",
             "Pothole", "Illegal Dumping",
             "Sidewalk Repair Issue", "Street Light Out")

get_done <- get_done %>%
  filter(service_name %in% keepers)

# ------- fun with quanteda -----

# create a quanteda::corpus object
# corpus() likes specific naming schemes for a document and associated text
# both need to be characters
get_done_corp <- get_done %>% 
  mutate_if(is.numeric, as.character) %>% 
  rename(doc_id = service_request_id,
         text = public_description) %>% 
  corpus()

get_done_corp
summary(get_done_corp)

# see quanteda Quick Start Guide -> Corpus Principles
# https://quanteda.io/articles/quickstart.html
# corpus should be as is. no processing, etc.

# extract the texts from the corpus
texts(get_done_corp)[1]
texts(get_done_corp)[2]

# several options available to automatically remove certain types of tokens
tokens(get_done_corp,
       remove_numbers = TRUE,
       remove_punct = TRUE,
       remove_symbols = TRUE,
       remove_twitter = TRUE)[1:5]

# alternatively, we can go straight to a matrix and have the options passed to
#  tokens()
# dfm() creates a document-feature matrix
get_done_dfm <- dfm(get_done_corp,
                    tolower = TRUE,
                    remove = stopwords("english"),
                    # below args passed to tokens()
                    remove_numbers = TRUE,
                    remove_punct = TRUE,
                    remove_symbols = TRUE,
                    remove_twitter = TRUE)


get_done_dfm

get_done_dfm[1:10, 1:10]

# most common words
# graffiti' and '72 hour violation' should be easy to classify
topfeatures(get_done_dfm, 10)

textplot_wordcloud(get_done_dfm,
                   max_words = 200,
                   min_count = 5)

# tf-idf and feature co-occurence matrices
dfm_tfidf(get_done_dfm)[1:10, 1:10]
get_done_fcm <- fcm(get_done_dfm)[1:10, 1:10]


# -------- modeling -------

# remove features (terms) that only appear in one document
# this cuts the features down by about 1/2
get_done_trimmed <- dfm_trim(get_done_dfm, min_docfreq = 2)

set.seed(42)
train_index <- sample(nrow(get_done_trimmed),
                      size = 0.8 * nrow(get_done_trimmed))
train_dfm <- get_done_trimmed[train_index, ]
test_dfm <- get_done_trimmed[-train_index, ]

train_labels <- get_done %>% 
  slice(train_index) %>% 
  pull(service_name)

test_labels <- get_done %>% 
  slice(-train_index) %>% 
  pull(service_name)


# keras needs labels in a special form
# first recode to integers then to one hot matrix
# this is ugly but couldn't find a better way
train_labels_int <- numeric(length(train_labels))
for (i in seq_along(train_labels)) {
  
  if (train_labels[i] == "72 Hour Violation") train_labels_int[i] = 0
  if (train_labels[i] == "Graffiti Removal") train_labels_int[i] = 1
  if (train_labels[i] == "Pothole") train_labels_int[i] = 2
  if (train_labels[i] == "Illegal Dumping") train_labels_int[i] = 3
  if (train_labels[i] == "Sidewalk Repair Issue") train_labels_int[i] = 4
  if (train_labels[i] == "Street Light Out") train_labels_int[i] = 5
  
}

test_labels_int <- numeric(length(test_labels))
for (i in seq_along(test_labels)) {
  
  if (test_labels[i] == "72 Hour Violation") test_labels_int[i] = 0
  if (test_labels[i] == "Graffiti Removal") test_labels_int[i] = 1
  if (test_labels[i] == "Pothole") test_labels_int[i] = 2
  if (test_labels[i] == "Illegal Dumping") test_labels_int[i] = 3
  if (test_labels[i] == "Sidewalk Repair Issue") test_labels_int[i] = 4
  if (test_labels[i] == "Street Light Out") test_labels_int[i] = 5
  
}

train_labels_one_hot <- to_categorical(train_labels_int, num_classes = 6)
test_labels_one_hot <- to_categorical(test_labels_int, num_classes = 6)

# baseline: naive bayes
naive_bayes <- textmodel_nb(train_dfm, train_labels,
                           distribution = "multinomial")

naive_bayes
summary(naive_bayes)

nb_test_pred <- predict(naive_bayes, newdata = test_dfm)


tibble(truth = as.factor(train_labels),
       estimate = predict(naive_bayes)) %>% 
  accuracy(truth, estimate)

# start simple and work our way up
two_layer <- keras_model_sequential() %>% 
  layer_dense(units = 256,
              activation = "relu",
              input_shape = ncol(train_dfm)) %>% 
  layer_dense(units = 128,
              activation = "relu") %>% 
  layer_dense(units = 6,
              activation = "softmax")

two_layer %>% 
  compile(
    optimizer = "rmsprop",
    loss = "categorical_crossentropy",
    metrics = c("accuracy")
  )

two_layer_hist <- two_layer %>% 
  fit(
    train_dfm,
    train_labels_one_hot,
    epochs = 10,
    batch_size = 512,
    validation_split = 0.2
  )




