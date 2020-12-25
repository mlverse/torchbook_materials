
# https://crudata.uea.ac.uk/~timo/datapages/naoi.htm
# https://crudata.uea.ac.uk/cru/data/nao/nao.dat

# https://climatedataguide.ucar.edu/climate-data/hurrell-north-atlantic-oscillation-nao-index-station-based

# https://www.ncdc.noaa.gov/teleconnections/nao/

library(torch)
library(dplyr)
library(tidyr)
library(ggplot2)

# start 3 years later, in 1824
# last valid value is 2020-7
nao <- read_table2("data/nao/nao.dat", col_names = FALSE, na = "-99.99", skip = 3) %>%
  select(-X1, -X14) %>%
  as.matrix() %>%
  t() %>%
  as.numeric() %>%
  .[1:(length(.) - 5)] %>%
  tibble(x = seq.Date(from = as.Date("1824-01-01"), to = as.Date("2020-07-01"), by = "months"), y = .) 

nao %>% ggplot(aes(x, y)) + geom_line(size = 0.2)

mean(nao$y, na.rm = TRUE)
quantile(nao$y, na.rm = TRUE)

nao[is.na(nao)] <- mean(nao$y, na.rm = TRUE)

# dataset -----------------------------------------------------------------

n_timesteps <- 6

nao_dataset <- dataset(
  
  name = "nao_dataset",
  
  initialize = function(nao, n_timesteps, random_sample = FALSE) {
    
    self$nao <- nao$y
    self$n_timesteps <- n_timesteps
    self$random_sample <- random_sample
    
  },
  
  .getitem = function(i) {
    
    if (self$random_sample == TRUE) {
      i <- sample(1:self$.length(), 1)
    }
    
    x <- torch_tensor(self$nao[i:(n_timesteps + i - 1)])
    y <- torch_tensor(self$nao[(i + 1):(n_timesteps + i)])
    list(x = x, y = y)
  },
  
  .length = function() {
    length(self$nao) - n_timesteps + 1
  }
  
)

train_ds <- nao_dataset(nao, n_timesteps)
length(train_ds)
first <- train_ds$.getitem(1)
first$x
first$y

batch_size <- 8
train_dl <- train_ds %>% dataloader(batch_size = batch_size)
length(train_dl)

iter <- valid_dl$.iter()
b <- iter$.next()
b
