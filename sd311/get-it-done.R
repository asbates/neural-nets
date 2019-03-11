
library(here)
library(tidyverse)
library(tidytext)
library(keras)


get_done_raw <- read_csv(here("data", "get_it_done_2018_requests_datasd.csv"))

skimr::skim(get_done_raw)


# remove rows with NA
get_done <- get_done_raw %>%
  filter(!is.na(public_description), !is.na(service_name))


# is their a relationship between empty descriptions and type of request?
get_done_missing <- get_done_raw %>%
  filter(is.na(public_description))

# no.
# but it looks like people select a category and then have the option
# to add a description
# check this out by downloading the app (but don' file a complaint!)
get_done_missing %>%
  count(service_name, sort = TRUE)


# for now let's just look at the description and category
# as noted above predicting category based on description is silly
# because looks like we have to choose a category first
# but let's pretend that isn't the case

get_done <- get_done %>%
  select(service_request_id, service_name, public_description)

get_done %>%
  count(service_name, sort = TRUE) %>%
  View()

# there are a lot of 'Other'
# maybe we could use the text to better classify the complaint?
# something like break Other into different categories
# but this would require manually labeling each Other and that
# would be way too time consuming
# in any case, since 'Other' is in the top 3 they probably need to
# come up with some more categories

# there are almost 400 categories.
# this is way too many to do the original idea on
# let's just pick to top few
# maybe the top 10, excluding other?

# actually, let's just keep the those that have more than 5000 entries
# besides 'Other'
keepers <- c("72 Hour Violation", "Graffiti Removal",
             "Pothole", "Illegal Dumping",
             "Sidewalk Repair Issue", "Street Light Out")

get_done <- get_done %>%
  filter(service_name %in% keepers)

# make sure we didn't make any typos
get_done %>%
  count(service_name, sort = TRUE)


get_done %>% View()

# ok, now we need to process the text
# let's save that for a later date b/c that might be a bit involved


# ================================
# ===== process the text =========
# ================================

get_done


get_done %>% 
  sample_n(100) %>% 
  unnest_tokens(word, public_description) %>% 
  anti_join(stop_words) %>% 
  #spread(service_request_id, word) %>% 
  View()


get_done %>% 
  unnest_tokens(sentence, public_description, token = "sentences") %>% 
  View()

set.seed(42)
no_stops <- get_done %>% 
  sample_n(100) %>% 
  mutate(description = str_remove(public_description, 
                                  "[:punct:]{1,}")) %>% 
  mutate(description = str_remove(description, "[:digit:]{1,}")) %>% 
  mutate(has_period = str_detect(description, "\\.")) %>% View()
  mutate(description = tolower(description)) %>% 
  mutate(description = map_chr(description,
                               ~.[!.%in% stop_words$word])) %>% 
  mutate(same = public_description == description) %>% 
  View()



get_done %>% 
  sample_n(100) %>% 
  mutate(n_words = length(public_description)) %>% 
  View()






