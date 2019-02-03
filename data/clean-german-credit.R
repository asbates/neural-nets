library(readr)
library(dplyr)

raw <- read_table("https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data-numeric",
                col_names = letters[1:25])

raw

# recode 1/2 as 0/1
# 1 means bad and 0 means good
# the uci website doesn't really say what the last column is or
#  how it's encoded 
#  but see https://www4.stat.ncsu.edu/~boos/var.select/german.credit.html
clean <- raw %>% 
  mutate(bad_loan = ifelse(y == 2, 0, y))

# this caused the *perfect* classification problem!
# clean %>% select (y, bad_loan) 

clean <- clean %>% 
  select(-y) # original response


write_csv(clean, here::here("data", "german-credit.csv"))
