
library(AmesHousing)
library(tidyverse)
library(recipes)

ames <- make_ames()


# create a simplified version with only numeric columns
ames_numeric <- ames %>% 
  select_if(is.numeric)

ames_num_rec <- recipe(Sale_Price ~., data = ames_numeric) %>% 
  step_center(all_predictors()) %>% 
  step_scale(all_predictors()) %>% 
  step_log(all_outcomes())

ames_numeric_processed <- ames_num_rec %>% 
  prep() %>% 
  bake(new_data = ames_numeric) %>% 
  rename(Log_Sale_Price = Sale_Price)

write_csv(ames_numeric_processed, "data/ames-numeric.csv")


# make a processed version of the entire data set
ames_rec <- recipe(Sale_Price ~., data = ames) %>% 
  step_dummy(all_nominal()) %>% 
  step_center(all_predictors()) %>% 
  step_scale(all_predictors()) %>% 
  step_log(all_outcomes())

ames_processed <- ames_rec %>% 
  prep() %>% 
  bake(new_data = ames) %>% 
  rename(Log_Sale_Price = Sale_Price) %>% 
  select(-Overall_Cond_Very_Excellent) # gets all set to NaN

write_csv(ames_processed, "data/ames-processed.csv")

