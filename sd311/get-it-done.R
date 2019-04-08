

library(here)
library(tidyverse)
library(quanteda)
library(keras)
library(yardstick)
library(rsample)


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

# what does distribution of issues look like
get_done %>% 
  count(service_name, sort = TRUE) %>% 
  mutate(prop = 100 * n / sum(n))

# prep work for later use
# keras needs factors to have numeric encoding starting with 0
get_done <- get_done %>% 
  mutate(service_name = fct_infreq(service_name),
         service_numeric = as.numeric(service_name) - 1)



# =======================================
# ====== overview of quanteda ===========
# =======================================

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


get_done_dfm # 24k words

get_done_dfm[1:10, 1:10]

# most common words
# graffiti' and '72 hour violation' should be easy to classify
topfeatures(get_done_dfm, 10)

# street graffiti   parked reported      car       st  vehicle 
# 14248    14190    13928    10168     8291     6882     6781 
# sidewalk    weeks    front 
# 6516     5810     5315 

textplot_wordcloud(get_done_dfm,
                   max_words = 200,
                   min_count = 5)

# tf-idf and feature co-occurence matrices
dfm_tfidf(get_done_dfm)[1:10, 1:10]
get_done_fcm <- fcm(get_done_dfm)[1:10, 1:10]


# ============================
# ===== modeling set up ======
# ============================

# remove features (terms) that only appear in one document
# this cuts the features down by about 1/2
get_done_trimmed <- dfm_trim(get_done_dfm, min_docfreq = 2)

get_done_trimmed  # 12k words

# split into train/test according to issue type
#  so that classes are balanced across train/test sets
# just using 'initial_split' to get indices because we really need to 
#  split the dfm 'get_done_trimmed' but it doesn't have the response 
set.seed(42)
split_obj <- initial_split(get_done, strata = "service_name")
train_index <- split_obj$in_id

train_dfm <- get_done_trimmed[train_index, ]
test_dfm <- get_done_trimmed[-train_index, ]

train_labels <- get_done %>% 
  slice(train_index) %>% 
  select(service_name, service_numeric)

test_labels <- get_done %>% 
  slice(-train_index) %>% 
  select(service_name, service_numeric)


# keras needs labels in a special form
train_labels_one_hot <- to_categorical(train_labels$service_numeric,
                                       num_classes = 6)
test_labels_one_hot <- to_categorical(test_labels$service_numeric,
                                      num_classes = 6)
# =======================
# ===== modeling ========
# =======================

# baseline: naive bayes via quanteda
naive_bayes <- textmodel_nb(train_dfm, train_labels$service_name,
                           distribution = "multinomial")

naive_bayes
summary(naive_bayes)

# training set accuracy
tibble(truth = train_labels$service_name,
       estimate = predict(naive_bayes)) %>% 
  accuracy(truth, estimate)   # 95.7



tibble(truth = test_labels$service_name,
       estimate = predict(naive_bayes, newdata = test_dfm)) %>% 
  accuracy(truth, estimate)  # 94.7

# start simple and work our way up if needed
# first use single hidden layer network
one_layer <- keras_model_sequential() %>% 
  layer_dense(units = 512,
              activation = "relu",
              input_shape = ncol(train_dfm)) %>% 
  layer_dense(units = 6,
              activation = "softmax")

one_layer %>% 
  compile(
    optimizer = "rmsprop",
    loss = "categorical_crossentropy",
    metrics = c("accuracy")
  )

# just train for 30 epochs in the interest of time
# and naive bayes did well so probably don't need to train that long anyways
one_layer_hist <- one_layer %>% 
  fit(
    train_dfm,
    train_labels_one_hot,
    epochs = 30,
    batch_size = 512,
    validation_split = 0.2
  )

# non-interactive ggplot
plot(one_layer_hist)

# validation accuracy drops right away
# let's re-train with 2 epochs
one_layer_short <- keras_model_sequential() %>% 
  layer_dense(units = 512,
              activation = "relu",
              input_shape = ncol(train_dfm)) %>% 
  layer_dense(units = 6,
              activation = "softmax")

one_layer_short %>% 
  compile(
    optimizer = "rmsprop",
    loss = "categorical_crossentropy",
    metrics = c("accuracy")
  )

one_layer_short %>% 
  fit(
    train_dfm,
    train_labels_one_hot,
    epochs = 2,
    batch_size = 512
  )

one_layer_short %>% 
  evaluate(train_dfm, train_labels_one_hot)
# $loss
# [1] 0.08755
# 
# $acc
# [1] 0.9758066

one_layer_short %>% 
  evaluate(test_dfm, test_labels_one_hot)

# $loss
# [1] 0.1482173
# 
# $acc
# [1] 0.9573975

# barely better than naive bayes


# add some complexity with one more dense layer
# this is probably unnecessary. it's really just for fun
# we will also use dropout
two_layer <- keras_model_sequential() %>% 
  layer_dense(units = 256,
              activation = "relu",
              input_shape = ncol(train_dfm)) %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 128,
              activation = "relu") %>% 
  layer_dropout(rate = 0.5) %>% 
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
    epochs = 30,
    batch_size = 512,
    validation_split = 0.2
  )

plot(two_layer_hist)

# similar to before model starts to overfit right away
two_layer_short <- keras_model_sequential() %>% 
  layer_dense(units = 256,
              activation = "relu",
              input_shape = ncol(train_dfm)) %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 128,
              activation = "relu") %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 6,
              activation = "softmax")

two_layer_short %>% 
  compile(
    optimizer = "rmsprop",
    loss = "categorical_crossentropy",
    metrics = c("accuracy")
  )

two_layer_short %>% 
  fit(
    train_dfm,
    train_labels_one_hot,
    epochs = 4,
    batch_size = 512
  )

two_layer_short %>% evaluate(train_dfm, train_labels_one_hot)
# $loss
# [1] 0.07588009
# 
# $acc
# [1] 0.9782203

two_layer_short %>% evaluate(test_dfm, test_labels_one_hot)
# $loss
# [1] 0.1522364
# 
# $acc
# [1] 0.9586421

# again, this is basically the same as naive bayes
# takeaway: neural networks are overkill for this problem


