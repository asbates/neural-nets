
library(AmesHousing)
library(tidyverse)
library(recipes)

ames <- make_ames()

ames
str(ames)
names(ames)

# create a simplified version with only numeric columns
ames_numeric <- ames %>% 
  select_if(is.numeric)

write_csv(ames_numeric, "data/ames-numeric.csv")


# make a processed version of the entire data set
ames_rec <- recipe(Sale_Price ~., data = ames) %>% 
  step_dummy(all_nominal()) %>% 
  step_center(all_predictors()) %>% 
  step_scale(all_predictors())

ames_processed <- ames_rec %>% 
  prep() %>% 
  bake(new_data = ames)

write_csv(ames_processed, "data/ames-processed.csv")

