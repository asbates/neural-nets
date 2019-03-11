

library(here)
library(tidyverse)
library(keras)


get_done_raw <- read_csv(here("data", "get_it_done_2018_requests_datasd.csv"))

# remove rows with NA
get_done <- get_done_raw %>%
  filter(!is.na(public_description), !is.na(service_name))

get_done <- get_done %>%
  select(service_request_id, service_name, public_description)

keepers <- c("72 Hour Violation", "Graffiti Removal",
             "Pothole", "Illegal Dumping",
             "Sidewalk Repair Issue", "Street Light Out")

get_done <- get_done %>%
  filter(service_name %in% keepers)



# take a subset for now until figure things out
set.seed(42)
get_done_small <- get_done %>% sample_n(100)



desc_tokenizer <- text_tokenizer(num_words = 500) # this is pretty slow
desc_tokenizer %>% fit_text_tokenizer(get_done_small$public_description)


desc_mat <- texts_to_matrix(desc_tokenizer, get_done_small$public_description,
                            mode = "count")

desc_mat_bin <- texts_to_matrix(desc_tokenizer, get_done_small$public_description,
                                   mode = "binary")

desc_tokenizer$word_counts

desc_seq <- text_to_word_sequence()











