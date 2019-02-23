
# this is a simple script just to show some code running on AWS

install.packages("tidyverse")
library(tidyverse)

data(starwars)

View(starwars)

starwars %>% 
  group_by(species) %>% 
  summarise(
    height = mean(height),
    mass = mean(mass)
  ) %>% 
  arrange(height, mass)



