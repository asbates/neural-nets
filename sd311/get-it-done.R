

library(here)
library(tidyverse)
library(quanteda)
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


# rm(get_done_raw)

# take a subset for now until figure things out
set.seed(42)
get_done_small <- get_done %>% sample_n(100)

# need to specify a document id and text column to corpus()
# both need to be characters
get_done_corp <- get_done_small %>% 
  mutate_if(is.numeric, as.character) %>% 
  rename(doc_id = service_request_id,
         text = public_description)

get_done_corp <- corpus(get_done_corp)

get_done_corp
summary(get_done_corp)

# see quanteda Quick Start Guide -> Corpus Principles
# https://quanteda.io/articles/quickstart.html
# corpus should be as is. no processing, etc.

# extract the texts from the corpus
texts(get_done_corp)

# plot summary information. e.g. number of Tokens in each document

corp_summary <- summary(get_done_corp)

# this isn't very good. really should sort
corp_summary %>% 
  ggplot(aes(Text, Tokens)) +
  geom_col() +
  facet_wrap(~service_name)

# looks like a number of people use @ for 'at'
# maybe replace @ with 'at'?
# for now let's just remove them since at is likely a stopword anyways
get_done$public_description[str_detect(get_done$public_description, "@")]

# the quick start guide recommends (or implies) most people don't want to
# use `tokens()`
# but i think we do in this case

# play around with these options on a few specific examples
tokens(get_done_corp,
       remove_numbers = TRUE,
       remove_punct = TRUE,
       remove_symbols = TRUE,
       remove_twitter = TRUE)[1:5]




# can pass all these options to `dfm()` as well
# but how to get from corpus -> tokenized -> dfm ?

# wait, `x` in `dfm(x, ......)` can be of class "tokens"

# this isn't right
get_done_small %>% 
  mutate(tokenized = tokens(get_done_corp,
                            remove_numbers = TRUE,
                            remove_punct = TRUE,
                            remove_symbols = TRUE,
                            remove_twitter = TRUE)) %>% 
  View()

tokenized <- tokens(get_done_corp,
                    remove_numbers = TRUE,
                    remove_punct = TRUE,
                    remove_symbols = TRUE,
                    remove_twitter = TRUE)
class(tokenized)

# for now let's just make a dfm and specify the `tokens` arguments

get_done_dfm <- dfm(get_done_corp,
                    stem = TRUE,
                    remove = stopwords("english"),
                    remove_numbers = TRUE,
                    remove_punct = TRUE,
                    remove_symbols = TRUE,
                    remove_twitter = TRUE)

# this is slow b/c size. don't do this get_done_dfm %>% View()

get_done_dfm
class(get_done_dfm)

# most common words
topfeatures(get_done_dfm, 10)

textplot_wordcloud(get_done_dfm)

# based on these two things 'graffiti' and '72 hour violation' should be easy

head(get_done_dfm[, 1:50]) %>% View()
rownames(get_done_dfm) # noooo!

# matrix looks sort of diagonal
# e.g. ones in first row then ones in second (diagonally)

# see what tfidf looks like
get_done_tfidf <- dfm_tfidf(get_done_dfm)

head(get_done_tfidf[, 1:50]) %>% View()

# see `text_model_nb` for use as a baseline

# feature names. i.e. all the words
featnames(get_done_dfm)

# now that have feature matrix, need to take this sparse representation
#  and make it less sparse
# look at h2o's word2vec
# i think the only other (reasonable) option is a keras embedding layer

# see also keras' tokenizer



