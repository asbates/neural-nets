
# this was adapted from
# https://keras.rstudio.com/articles/tutorial_basic_regression.html


library(keras)
library(tidyverse)
library(rsample)
library(yardstick)

# simplified & processed version of ames housing data set
# only numeric variables are included
# all predictors are centered and scaled
# outcome: sale price is log transformed
ames <- read_csv("data/ames-numeric.csv")

# --------------------------------------
# basic example of fitting a keras model
# --------------------------------------


# initial training/testing split
ames_split <- initial_split(ames)
ames_train <- training(ames_split)
ames_test <- testing(ames_split)


# keras takes x,y arguments
train_x <- ames_train %>% 
  select(-Log_Sale_Price) %>% 
  as.matrix()

train_y <- ames_train %>% 
  pull(Log_Sale_Price)

test_x <- ames_test %>% 
  select(-Log_Sale_Price) %>% 
  as.matrix()


model <- keras_model_sequential() %>% 
  layer_dense(units = 10, 
              activation = "sigmoid",
              input_shape = ncol(train_x)) %>% 
  layer_dense(units = 10,
              activation = "sigmoid") %>% 
  layer_dense(units = 1)

model %>% compile(
  loss = "mse",
  optimizer = optimizer_sgd(),
  metrics = "mse"
)

# fit the model
# as is, this builds an interactive plot showing training loss
# see the reference at the top for how to adjust this
#  and how to determine when to stop training, set an early stopping
#  criteria, etc.
model %>% fit(
  train_x,
  train_y,
  epochs = 10
)

pred <- model %>% 
  predict(test_x) %>% 
  as.vector()

ames_test %>% 
  mutate(predicted_price = pred) %>% 
  metrics(Log_Sale_Price, predicted_price)


# ------------------------
# a more detailed example
# ------------------------

# let's pretend we didn't make predictions on the test set yet
# we will use cross validation to see how to tune hyperparameters

# to keep running time low, we will simplify by:
#  using two models: one with 2 hidden layers and one with 3
#  only using 2 folds of cross validation

# set up CV folds
ames_cv <- vfold_cv(ames_test, v = 2)


# we need to put everything above into a function we can apply to the 
#  ames_cv object

# sets up a model with either two or 3 hidden layers
set_model <- function(two_layers){
  if (two_layers){
    model <- keras_model_sequential() %>% 
      layer_dense(units = 10, 
                  activation = "sigmoid",
                  input_shape = ncol(train_x)) %>% 
      layer_dense(units = 10,
                  activation = "sigmoid") %>% 
      layer_dense(units = 1)
  } else{
    model <- keras_model_sequential() %>% 
      layer_dense(units = 10, 
                  activation = "sigmoid",
                  input_shape = ncol(train_x)) %>% 
      layer_dense(units = 10,
                  activation = "sigmoid") %>%
      layer_dense(units = 10,
                  activation = "sigmoid") %>% 
      layer_dense(units = 1)
  }
  model
}


# Display training progress by printing a single dot for each completed epoch.
# taken directly from the reference above
print_dot_callback <- callback_lambda(
  on_epoch_end = function(epoch, logs) {
    if (epoch %% 80 == 0) cat("\n")
    cat(".")
  }
)

fit_model <- function(splits, two_layers, ...) {
  
  # set up x,y
  train_x <- analysis(splits) %>% 
    select(-Log_Sale_Price) %>% 
    as.matrix()
  
  train_y <- analysis(splits) %>% 
    pull(Log_Sale_Price)
  
  test_x <- assessment(splits) %>% 
    select(-Log_Sale_Price) %>% 
    as.matrix()
  
  model <- set_model(two_layers)
  
  model %>% compile(
    loss = "mse",
    optimizer = optimizer_sgd(),
    metrics = "mse"
  )
  
  model %>% fit(
    train_x,
    train_y,
    epochs = 10,
    verbose = 0,
    callbacks = list(print_dot_callback)
  )
  
  pred <- model %>% 
    predict(test_x) %>% 
    as.vector()
  
  analysis(splits) %>% 
    mutate(predicted_price = pred) %>% 
    metrics(Log_Sale_Price, predicted_price)
  
}

ames_cv_results <- ames_cv %>% 
  mutate(model_two_layers = map(.$splits,fit_model, two_layers = TRUE)) %>% 
  mutate(model_three_layers = map(.$splits,fit_model, two_layers = FALSE))

# extract the errors and average across the folds
mean_error <- function(df, ...){
  df %>% 
  select(...) %>% 
  unnest() %>% 
    group_by(.metric) %>% 
    summarise(mean = mean(.estimate))
}

# mean CV error for model with 2 hidden layers
ames_cv_results %>% 
  mean_error(model_two_layers)

# mean CV error for model with 3 hidden layers
ames_cv_results %>% 
  mean_error(model_three_layers)
