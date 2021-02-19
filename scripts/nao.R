# https://otexts.com/fpp3

# https://crudata.uea.ac.uk/~timo/datapages/naoi.htm
# https://crudata.uea.ac.uk/cru/data/nao/nao.dat

# https://climatedataguide.ucar.edu/climate-data/hurrell-north-atlantic-oscillation-nao-index-station-based

# https://www.ncdc.noaa.gov/teleconnections/nao/

library(torch)
library(tidyverse)
library(fable)
library(tsibble)
library(feasts)
library(tsibbledata)
# vic_elec

# start 3 years later, in 1824
# last valid value is 2020-7
nao <-
  read_table2(
    "data/nao/nao.dat",
    col_names = FALSE,
    na = "-99.99",
    skip = 3
  ) %>%
  select(-X1,-X14) %>%
  as.matrix() %>% 
  t() %>%
  as.vector() %>%
  .[1:(length(.) - 5)] %>%
  tibble(x = seq.Date(
    from = as.Date("1824-01-01"),
    to = as.Date("2020-07-01"),
    by = "months"
  ),
  y = .) %>%
  mutate(x = yearmonth(x)) %>%
  fill(y) %>%
  as_tsibble(index = x) 
  
nrow(nao)

nao_train <- nao %>% filter(x <  yearmonth("1990-01"))
nao_valid <- nao %>% filter(x >=  yearmonth("1990-01"))



# analysis ----------------------------------------------------------------


# STL
# season(window=13)
# trend(window=13)
cmp <- nao_valid %>%
  model(STL(y)) %>%
  components()
cmp %>% autoplot()

cmp <- nao_valid %>%
  model(STL(y ~ season(window = 7))) %>%
  components()
cmp %>% autoplot()

nao_valid %>% features(y, feat_stl)
feat_stl(nao_valid$y, .period = 12, s.window = 7) %>% round(2)

# ACF
nao_valid %>% features(y, feat_acf)
nao_valid %>% ACF(y) %>% autoplot()

# other features
# rate at which autocorrelations decrease as the lag between pairs of values increases
# > 0.5: long-term positive autocorrelations
# < 0.5: mean-reverting
# 0.5: random walk
nao_valid %>% features(y, coef_hurst) 
nao_valid %>% features(y, feat_spectral) #[0, 1]
nao_valid %>% features(y, feat_acf)
nao_valid %>% features(y, feat_acf)
nao_valid %>% features(y, feat_acf)
  

# fit

fit <- nao_train %>% model(
  # Error ={A,M}, Trend ={N,A,Ad} and Seasonal ={N,A,M}.
  ets = ETS(y ~ season(method = "A", gamma = 0.5))
)
fit

fit <- nao_train %>% model(
  ets = ETS(y ~ season(method = "A", gamma = 0.1)), # 0: seasonal pattern will not change
  ets2 = ETS(y ~ season(method = "A", gamma = 0.5)),
  ets3 = ETS(y ~ season(method = "A", gamma = 0.9)), # 1: seasonality will have no memory of past periods
  arima = ARIMA(y),
  snaive = SNAIVE(y)
) 

fc <- fit %>%
  forecast(h = "2 years") 

fc %>% 
  autoplot(filter(nao_valid, x < yearmonth("1992-01")), level = NULL)

accuracy(fc, filter(nao_valid, x < yearmonth("1992-01")))

fit %>% select(ets3) %>% report()
fit %>% report()

# dataset -----------------------------------------------------------------

n_timesteps <- 12

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
    
    x <- torch_tensor(self$nao[i:(n_timesteps + i - 1)])$unsqueeze(2)
    y <- torch_tensor(self$nao[n_timesteps + i])
    list(x = x, y = y)
  },
  
  .length = function() {
    length(self$nao) - n_timesteps
  }
  
)

train_ds <- nao_dataset(nao_train, n_timesteps, random_sample = TRUE)
length(train_ds)
# first <- train_ds$.getitem(1)
# first$x
# first$y

batch_size <- 32
train_dl <- train_ds %>% dataloader(batch_size = batch_size)
length(train_dl)

# iter <- dataloader_make_iter(train_dl)
# b <- iter %>% dataloader_next()
# b

valid_ds <- nao_dataset(nao_valid, n_timesteps)
#length(valid_ds)
valid_dl <- valid_ds %>% dataloader(batch_size = batch_size)
length(valid_dl)


# model -------------------------------------------------------------------


model <- nn_module(
  
  initialize = function(type, hidden_size) {
    
    self$type <- type
    
    self$rnn <- if (self$type == "gru") {
      nn_gru(
        input_size = 1,
        hidden_size = hidden_size,
        num_layers = 1,
        #dropout = 0.2,
        batch_first = TRUE
      )
    } else {
      nn_lstm(
        input_size = 1,
        hidden_size = hidden_size,
        num_layers = 1,
        #dropout = 0.2,
        batch_first = TRUE
      )
    }
    
    self$output <- nn_linear(hidden_size, 1)
    
  },
  
  forward = function(x) {
    
    # for each layer, hidden state for t = seq_len
    x <- self$rnn(x)
    x <- if (self$type == "gru")x[[2]] else x[[2]][[1]]
     
    x <- x$squeeze()
    x %>% self$output() 
    
  }
  
)

device <- torch_device(if (cuda_is_available()) "cuda" else "cpu")
#device <- "cpu"

net <- model("gru", 32)

net <- net$to(device = device)
#net

#net(b$x$to(device = device))

# train -------------------------------------------------------------------

optimizer <- optim_adam(net$parameters, lr = 0.001)

num_epochs <- 100

train_batch <- function(b) {
  
  optimizer$zero_grad()
  output <- net(b$x$to(device = device))
  target <- b$y$to(device = device)
  
  loss <- nnf_mse_loss(output, target)
  
  if (i %% 11111 == 0) {
  
    print(as.matrix(output$to(device = "cpu")))
    print(as.matrix(target$to(device = "cpu")))
  }
  
  i <<- i + 1

  loss$backward()
  optimizer$step()
  
  #gc(full = TRUE)
  
  loss$item()
  
}

valid_batch <- function(b) {
  
  output <- net(b$x$to(device = device))
  target <- b$y$to(device = device)
  
  loss <- nnf_mse_loss(output, target)
  
  loss$item()
  
}

for (epoch in 1:num_epochs) {
  
  net$train()
  train_loss <- c()
  
  i <<- 1
  
  coro::loop(for (b in train_dl) {
   loss <-train_batch(b)
   train_loss <- c(train_loss, loss)
  })
  
  #torch_save(net, paste0("model_", epoch, ".pt"))
  cat(sprintf("\nEpoch %d, training: loss: %3.3f \n", epoch, mean(train_loss)))
  
  net$eval()
  valid_loss <- c()

  coro::loop(for (b in valid_dl) {
    loss <- valid_batch(b)
    valid_loss <- c(valid_loss, loss)
  })

  cat(sprintf("\nEpoch %d, validation: loss: %3.3f \n", epoch, mean(valid_loss)))
}
