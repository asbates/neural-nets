
# clean up data from
# https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/
# and write to csv

# note: the file was downloaded from the website above
# and then deleted b/c I don't need a .xsl file

library(readxl)
library(dplyr)
library(readr)
library(here)

file_path <- "/Users/andrewbates/Desktop/Concrete_Data.xls"
columns <- c("cement", "slag", "ash", "water", "superplast",
             "coarse", "fine", "age", "strength")
raw <- read_xls(file_path, 
                col_names = columns,
                skip = 1)

raw

# looks good as is so go ahead and write it
write_csv(raw, here("data", "concrete-strength.csv"))



