library(tidyverse)
library(lubridate)
library(tsibble)
library(feasts)
library(tsibbledata)


vic_elec_2014 <-  vic_elec %>%
  filter(year(Date) == 2014) %>%
  select(-c(Date, Holiday)) %>%
  mutate(Demand = scale(Demand), Temperature = scale(Temperature)) %>%
  pivot_longer(-Time, names_to = "variable") %>%
  update_tsibble(key = variable)

vic_elec_2014 %>% autoplot() + 
  scale_colour_manual(values = c("#08c5d1", "#ffbf66")) +
  theme_minimal()

vic_elec_2014 %>% filter(month(Time) == 1) %>% 
  autoplot() + 
  scale_colour_manual(values = c("#08c5d1", "#ffbf66")) +
  theme_minimal()

vic_elec_2014 %>% filter(month(Time) == 7) %>% 
  autoplot() + 
  scale_colour_manual(values = c("#08c5d1", "#ffbf66")) +
  theme_minimal()

# explore ----------------------------------------------------------------

vic_elec_2014 <-  vic_elec %>%
  filter(year(Date) == 2014) %>%
  select(-c(Date, Holiday))

cmp <- vic_elec_2014 %>% 
  model(STL(Demand)) %>%
  components()
cmp %>% autoplot()

cmp <- vic_elec_2014 %>% filter(month(Time) == 7) %>%
  model(STL(Demand)) %>% 
  components()
cmp %>% autoplot()

cmp <- vic_elec_2014 %>% filter(month(Time) == 1) %>%
  model(STL(Demand)) %>% 
  components()
cmp %>% autoplot()
